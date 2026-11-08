"""Synthetic same-date category aggregation; no encoder downloads or server data."""

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"forecasting_task"))
from forecasting_task import run
from forecasting_task.models.CategoryText import CategoryTextModel, scorer_config
from forecasting_task.models.GroupText import ContextNoveltyTextModel, TEXT_GROUPS, build_text_model
from forecasting_task.category_data import CATEGORIES, FILING, PROVENANCE, file_hash, reproduce_groups, load_manifest
from forecasting_task.data_provider.category import CategoryDataset
from forecasting_task.data_provider.dataset import Dataset_Custom
from forecasting_task.gate_diagnostics import write_gate_validation
from prepare_category_embeddings import build_ticker, clean_texts
import test_context_novelty as fixtures


def model(pool):
    return CategoryTextModel(pool, hidden_size=64, pred_len=3, num_layers=1, group_gate="context_price_only")


def batch(n=3):
    values = torch.nn.functional.normalize(torch.randn(n, 4, 5, 384), dim=-1)
    mask = torch.ones(n, 4, 5, dtype=torch.bool)
    mask[:, 2:, 3:] = False
    mask[:, 1, 1:] = False
    mask[0, 0] = False
    values[~mask] = 0
    x, price, grouped = torch.randn(n, 6, 4), torch.randn(n, 3, 4), torch.randn(n, 5, 384)
    return x, price, grouped, dict(category_embeddings=values, category_present=mask, filing_present=torch.ones(n, dtype=torch.bool))


def make_data(root):
    dates = [*pd.date_range("2021-12-01", periods=18), *pd.date_range("2022-01-03", periods=12), *pd.date_range("2023-01-03", periods=12)]
    meta = pd.DataFrame({"date": [str(d.date()) for d in dates], "ticker": ["SYN"]*42,
                         **{name: np.arange(42, dtype=float)+i for i, name in enumerate(("open", "high", "low", "close"))}})
    columns = [c for cols in CATEGORIES.values() for c in cols]+FILING
    texts = pd.DataFrame({c: [" summary "]*42 for c in columns})
    texts.loc[:15, "macro_category1"] = " "
    texts.loc[:, "sector_category2"] = None
    clean = clean_texts(texts, columns)
    rng = np.random.default_rng(7)
    embeddings = {}
    for column in columns:
        e = rng.normal(size=(42,384)).astype(np.float32)
        e /= np.linalg.norm(e, axis=-1, keepdims=True)
        e[clean[column].isna()] = 0
        embeddings[column] = e
    parts = [meta]
    for group, names in {**CATEGORIES, "filing": FILING}.items():
        total = np.zeros((42,384), np.float32)
        count = np.zeros(42, np.int32)
        for column in names:
            total += embeddings[column]
            count += clean[column].notna().to_numpy()
        mean = np.zeros_like(total)
        np.divide(total,count[:,None],out=mean,where=count[:,None]>0)
        parts.append(pd.DataFrame(mean, columns=[f"{group}_emb{i}" for i in range(384)]))
    legacy = pd.concat(parts,axis=1)
    frame, maximum = build_ticker(meta, clean, embeddings, legacy, 1e-6)
    root.mkdir()
    frame.to_parquet(root/"SYN.parquet",index=False)
    manifest = {"schema":"fintexts_categories_v1","provenance":PROVENANCE.copy(),"categories":CATEGORIES,"filing_columns":FILING,
                "encoder_versions":{k:"synthetic" for k in ("sentence-transformers","transformers","torch","datasets")},"encoder_max_seq_length":256,"atol":1e-6,
                "verification":{"status":"passed"},"files":{"SYN":{"rows":42,"sha256":file_hash(root/"SYN.parquet"),
                 "source_text_sha256":"a"*64,"legacy_sha256":"b"*64,"max_abs_diff":maximum}}}
    (root/"manifest.json").write_text(json.dumps(manifest))
    return frame, legacy, manifest


class CategoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)

    def test_masks_single_empty_normalization_identical_values_and_explicit_presence(self):
        for n in (1,3):
            x,p,e,meta=batch(n)
            text=model("attention")
            output,d=text(p,x[:,-1],e,price_window=x,return_details=True,**meta)
            self.assertEqual(output.shape,(n,3,4))
            torch.testing.assert_close(d["category_beta"].sum(-1),meta["category_present"].any(-1).float())
            self.assertTrue((d["category_beta"][~meta["category_present"]]==0).all())
            self.assertTrue((d["category_pooled"][0,0]==0).all())
            self.assertTrue((d["contributions"][0,0]==0).all())
            torch.testing.assert_close(d["category_beta"][:,1,0],torch.ones(n),rtol=0,atol=0)
            constant=torch.randn(n,4,1,384).expand(-1,-1,5,-1).clone()
            constant[~meta["category_present"]]=0
            changed={**meta,"category_embeddings":constant}
            _,other=text(p,x[:,-1],e,price_window=x,return_details=True,**changed)
            expected=constant[:,:,0]*meta["category_present"].any(-1)[...,None]
            torch.testing.assert_close(other["category_pooled"],expected)
        # Cancellation does not erase raw-text presence or a group-head bias.
        x,p,e,meta=batch(1)
        meta["category_present"][0,0,:2]=True
        meta["category_embeddings"][0,0,0]=1
        meta["category_embeddings"][0,0,1]=-1
        with torch.no_grad():
            text.group_mlps["macro"][0].bias.fill_(3)
        _,d=text(p,x[:,-1],e,price_window=x,return_details=True,**meta)
        self.assertTrue(d["present"][0,0])
        torch.testing.assert_close(d["category_pooled"][0,0],torch.zeros(384),rtol=0,atol=0)
        self.assertGreater(d["contributions"][0,0].abs().sum().item(),0)

    def test_permutation_and_no_group_or_batch_mixing_with_trained_scorer(self):
        text=model("attention")
        with torch.no_grad():
            text.category_scorer.v.normal_()
        x,p,e,meta=batch()
        a,d=text(p,x[:,-1],e,price_window=x,return_details=True,**meta)
        order=torch.tensor([4,0,3,2,1])
        perm={**meta,"category_embeddings":meta["category_embeddings"][:,:,order],"category_present":meta["category_present"][:,:,order]}
        b,other=text(p,x[:,-1],e,price_window=x,return_details=True,**perm)
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(d["category_beta"][:,:,order],other["category_beta"])
        torch.testing.assert_close(d["category_pooled"],other["category_pooled"])
        changed={**meta,"category_embeddings":meta["category_embeddings"].clone()}
        changed["category_embeddings"][1:,0]*=-2
        _,other=text(p,x[:,-1],e,price_window=x,return_details=True,**changed)
        torch.testing.assert_close(d["category_pooled"][0],other["category_pooled"][0],rtol=0,atol=0)
        torch.testing.assert_close(d["category_pooled"][:,1:],other["category_pooled"][:,1:],rtol=0,atol=0)

    def test_uniform_legacy_path_and_initial_attention_film_rng_and_batching(self):
        for seed in (7,17,27):
            reference=None
            for pool in ("uniform","attention"):
                run.set_seed(seed)
                film=fixtures.ContextNoveltyTests.make_film().eval()
                text=model(pool)
                state=film.state_dict(),text.group_mlps.state_dict(),text.gate.state_dict(),torch.get_rng_state().clone()
                x,p,e,meta=batch()
                a,d=text(p,x[:,-1],e,price_window=x,return_details=True,**meta)
                loader=DataLoader(TensorDataset(torch.arange(31)),batch_size=4,shuffle=True)
                run._configure_gate_sampling(loader,seed)
                orders=[torch.cat([b[0] for b in loader]) for _ in range(3)]
                if reference is None:
                    reference=state,a,orders
                    old=ContextNoveltyTextModel(64,3,1,group_gate="context_price_only")
                    old.load_state_dict(text.state_dict(),strict=True)
                    grouped=torch.cat((d["category_pooled"],e[:,4:5]),1)
                    legacy=old(p,x[:,-1],grouped,price_window=x,novelty=torch.zeros(3,5),pair_available=torch.zeros(3,5))
                    torch.testing.assert_close(a,legacy,rtol=0,atol=0)
                else:
                    for actual,expected in zip(state[:3],reference[0][:3]):
                        for key,value in actual.items():
                            torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
                    torch.testing.assert_close(state[3],reference[0][3],rtol=0,atol=0)
                    torch.testing.assert_close(a,reference[1],rtol=0,atol=0)
                    for actual,expected in zip(orders,reference[2]):
                        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                    self.assertEqual(text.category_scorer.W_c.in_features,8)
                    self.assertEqual(sum(v.numel() for v in text.category_scorer.parameters()),6304)

    def test_v_gradients_optimizer_and_component_decomposition(self):
        text=model("attention")
        film=fixtures.ContextNoveltyTests.make_film()
        optimizer=torch.optim.Adam([*film.parameters(),*text.parameters()],lr=.001)
        ids=[id(p) for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(ids),len(set(ids)))
        x,p,e,meta=batch()
        for step in range(2):
            optimizer.zero_grad()
            output,d=text(film(x,None,None,None),x[:,-1],e,price_window=x,return_details=True,**meta)
            output.square().mean().backward()
            self.assertGreater(text.category_scorer.v.grad.abs().sum().item(),0)
            if step==0:
                self.assertTrue((text.category_scorer.W_e.weight.grad==0).all())
                self.assertTrue((text.category_scorer.W_c.weight.grad==0).all())
            else:
                self.assertGreater(text.category_scorer.W_e.weight.grad.abs().sum().item(),0)
            optimizer.step()
        torch.testing.assert_close(d["category_contributions"].sum(2),d["contributions"][:,:4])
        self.assertTrue((d["category_contributions"][~meta["category_present"]]==0).all())
        self.assertTrue(all(ids.count(id(p))==1 for p in text.category_scorer.parameters()))

    def test_preparation_provenance_reproduction_and_dataset_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/"data"
            frame,old,manifest=make_data(root)
            self.assertTrue(all(v<=1e-6 for v in reproduce_groups(frame,1e-6).values()))
            for split in ("train","val","test"):
                base=Dataset_Custom(6,2,3,split,list(TEXT_GROUPS),root_path=str(root),data_path="SYN.parquet")
                data=CategoryDataset(base)
                np.testing.assert_array_equal(data[0][0],base[0][0])
                np.testing.assert_array_equal(data[0][1],base[0][1])
                row=5
                np.testing.assert_array_equal(data[0][6]["category_embeddings"],data.categories[row])
                self.assertEqual(data.window_end_date(0),data.row_dates[row])
            bad=frame.copy()
            bad.loc[0,"macro_emb0"]+=.01
            with self.assertRaises(ValueError):
                reproduce_groups(bad,1e-6)
            manifest["provenance"]["encoder_revision"]="bad"
            (root/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_manifest(root,"SYN")

    def test_small_server_verifier_preserves_legacy_and_rejects_tampering(self):
        from prepare_category_embeddings import verify
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            frame,old,manifest=make_data(root/"data")
            legacy=root/"legacy";legacy.mkdir()
            old.to_parquet(legacy/"SYN.parquet",index=False)
            original=(legacy/"SYN.parquet").read_bytes()
            manifest["files"]["SYN"]["legacy_sha256"]=file_hash(legacy/"SYN.parquet")
            (root/"data/manifest.json").write_text(json.dumps(manifest))
            with contextlib.redirect_stdout(io.StringIO()) as console:
                verify(root/"data",legacy,expected_tickers=1)
            self.assertIn("atol=1e-06",console.getvalue())
            self.assertEqual((legacy/"SYN.parquet").read_bytes(),original)
            frame.loc[0,"macro_category1_emb0"]+=.01
            frame.to_parquet(root/"data/SYN.parquet",index=False)
            with self.assertRaisesRegex(ValueError,"fingerprint"):
                verify(root/"data",legacy,expected_tickers=1)

    def test_runner_restore_best_checkpoint_and_saved_category_components(self):
        for pool in ("uniform","attention"):
            with self.subTest(pool=pool),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                make_data(root/"data")
                argv=["--model_type","FiLM","--group_gate","context_price_only","--category_pool",pool,
                      "--root_path",str(root/"data"),"--data_path","SYN.parquet","--seq_len","6","--label_len","2",
                      "--batch_size","4","--e_layers","1","--num_epoch","2","--code_commit","a"*40,"--logdir",str(root/"out")]
                from forecasting_task.data_provider.category import get_category_dataloader
                def factory(**kwargs):
                    data=CategoryDataset(Dataset_Custom(**{k:v for k,v in kwargs.items() if k!="batch_size"}))
                    train=kwargs["flag"]=="train"
                    return data,DataLoader(data,batch_size=kwargs["batch_size"],shuffle=train,drop_last=train)
                original=run._run_epoch
                epochs=[]
                def epoch(loader,dataset,price,text,args,device,optimizer=None,collect_io=False):
                    result=original(loader,dataset,price,text,args,device,optimizer,collect_io)
                    if optimizer is not None:
                        epochs.append(1)
                        if len(epochs)==2:
                            with torch.no_grad():
                                for head in text.group_mlps.values():
                                    head[0].bias.add_(1000)
                    return result
                with patch("forecasting_task.data_provider.category.get_category_dataloader",side_effect=factory), \
                     patch.object(run,"_run_epoch",side_effect=epoch),patch("torch.cuda.is_available",return_value=False), \
                     patch("forecasting_task.models.FiLM.device",torch.device("cpu")),contextlib.redirect_stdout(io.StringIO()):
                    run.main(argv)
                output=root/"out"
                args=yaml.safe_load((output/"args.yaml").read_text())
                cp=torch.load(output/"best_checkpoint.pt",weights_only=True)
                summary=json.loads((output/"summary.json").read_text())
                self.assertEqual(cp["epoch"],1)
                for source in (args,cp,summary):
                    self.assertEqual(source["category_pool"],pool)
                    self.assertEqual(source["category_scorer_config"],scorer_config(pool))
                with patch("forecasting_task.models.FiLM.device",torch.device("cpu")):
                    price=run.FiLM(SimpleNamespace(**args))
                text=build_text_model(args["text_model_config"],args["group_weight_mode"])
                price.load_state_dict(cp["model_state_dict"])
                text.load_state_dict(cp["text_model_state_dict"])
                data,loader=factory(seq_len=6,label_len=2,pred_len=3,flag="test",used_col_prefixes=list(TEXT_GROUPS),
                                    root_path=str(root/"data"),data_path="SYN.parquet",batch_size=4)
                with torch.no_grad():
                    predictions=original(loader,data,price,text,SimpleNamespace(**args),torch.device("cpu"),collect_io=True)
                torch.testing.assert_close(predictions[3],torch.load(output/"test_io.pt",weights_only=False)["outputs"],rtol=0,atol=0)
                for filename in ("validation_category_diagnostics.csv","validation_prediction_components.csv"):
                    with (output/filename).open(newline="") as handle:
                        rows=list(csv.DictReader(handle))
                    self.assertGreater(len(rows),0)
                    if "components" in filename:
                        for row in rows:
                            reconstructed=sum(float(row[k]) for k in ("film_component","last_price_component","text_component"))
                            self.assertAlmostEqual(reconstructed,float(row["prediction"]),places=5)
                    else:
                        self.assertEqual(len(rows),10*17)
                        self.assertTrue(all(row["source_column"] in [c for cols in CATEGORIES.values() for c in cols]+["filing_group_mean"] for row in rows))
                val,valloader=factory(seq_len=6,label_len=2,pred_len=3,flag="val",used_col_prefixes=list(TEXT_GROUPS),
                                     root_path=str(root/"data"),data_path="SYN.parquet",batch_size=4)
                with tempfile.TemporaryDirectory() as repeated:
                    diag_args=SimpleNamespace(**{**args,"logdir":repeated,"diagnostic_best_val_mse":summary["best_val_mse"]})
                    write_gate_validation(valloader,val,price,text,diag_args,torch.device("cpu"),run._prepare_batch)
                    self.assertEqual((Path(repeated)/"validation_category_diagnostics.csv").read_bytes(),(output/"validation_category_diagnostics.csv").read_bytes())
                if pool=="uniform":
                    import analyze_context_price_category_attention as analysis
                    old_config={k:v for k,v in args["text_model_config"].items() if k not in ("category_pool","category_scorer_config")}
                    old_args={**args,"text_model_config":old_config}
                    legacy_output=root/"legacy_run";legacy_output.mkdir()
                    torch.save({**cp,"args":old_args},legacy_output/"best_checkpoint.pt")
                    result=analysis.verify_old_uniform(analysis.Run("last","SYN",args["seed"],legacy_output),old_args,summary,root/"data")
                    self.assertEqual(result["status"],"passed")
