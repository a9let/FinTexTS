"""Matched metadata, truthful window verification and validation-only selection."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import analyze_context_price_category_attention as analysis
from forecasting_task.window_evidence import window_evidence


class Windows:
    seq_len,pred_len=64,3
    row_dates=pd.bdate_range("2022-01-03",periods=72).strftime("%Y-%m-%d").to_numpy()
    def __len__(self):
        return 2
    def __getitem__(self,index):
        p=np.arange(72*4,dtype=np.float32).reshape(72,4)/100
        return p[index:index+64],p[index+48:index+67]
    def window_end_date(self,index):
        return str(self.row_dates[index+63])


class AnalysisTests(unittest.TestCase):
    def fixture(self,root):
        roots=[root/name for name in ("new","last","baseline")]
        dataset=Windows()
        evidence={split:window_evidence(dataset) for split in ("val","test")}
        scalers={split:{"mean":np.arange(4,dtype=float),"scale":np.ones(4),"var":np.ones(4)} for split in evidence}
        cat={split:{"windows":2,"category_sha":"synthetic"} for split in evidence}
        for variant in analysis.VARIANTS:
            primary=variant in ("attention","uniform")
            root=roots[0] if primary else roots[1] if variant=="last" else roots[2]
            condition=variant if primary else "context_price_only" if variant=="last" else "baseline"
            for ticker in analysis.TICKERS:
                for seed_index,seed in enumerate(analysis.SEEDS):
                    output=root/"full"/condition/ticker/f"seed{seed}/output"
                    output.mkdir(parents=True)
                    args={**analysis.COMMON,"model_type":"FiLM","seed":seed,"data_path":ticker+".parquet",
                          "task_name":"short_term_forecast","e_layers":2,"output_attention":False,"enc_in":4,"dec_in":4,"c_out":4,
                          "group_gate":"context_price_only","fixed_alpha":.1,"text_weight_mode":"fixed","text_weight":.1,
                          "text_num_layers":1,"used_col_prefixes":analysis.GROUPS,"fusion_mode":"blend","group_weight_mode":"context_price_only",
                          "text_model_config":{"group_gate":"context_price_only"},"sampling_policy":"independent_train_generator",
                          "gate_structure":{"type":"Linear","in_features":18,"out_features":5,"parameters":95,"initialization":"zeros"},
                          "text_mlp_structure":[{"type":"Linear","in_features":384,"out_features":3}]}
                    summary={"seed":seed,"data_path":args["data_path"],"model":"FiLM","best_epoch":1,
                             "group_gate":"context_price_only","fixed_alpha":.1,"text_weight_mode":"fixed",
                             "best_val_mse":{"attention":.3,"uniform":.2,"last":.1,"original_baseline":.4}[variant]+seed_index*.05,
                             "best_test_mse":.1 if variant=="attention" else .3,"best_test_mae":.05}
                    if primary:
                        for source in (args,summary,args["text_model_config"]):
                            source.update(category_pool=variant,category_scorer_config=analysis.scorer_config(variant))
                        for source in (args,summary):
                            source.update(window_evidence=evidence,category_window_evidence=cat,category_manifest_sha256="verified_manifest")
                    (output/"summary.json").write_text(json.dumps(summary))
                    torch.save({"args":args,"epoch":1,"scaler_mean":[0.,1.,2.,3.],"scaler_scale":[1.]*4,"scaler_var":[1.]*4},output/"best_checkpoint.pt")
                    torch.save({"inputs":torch.from_numpy(np.stack([dataset[i][0] for i in range(len(dataset))])),
                                "ground_truths":torch.from_numpy(np.stack([dataset[i][1][-3:] for i in range(len(dataset))]))},output/"test_io.pt")
                    if variant!="original_baseline":
                        with (output/"validation_group_diagnostics.csv").open("w",newline="") as handle:
                            writer=csv.DictWriter(handle,fieldnames=["window_index","group","input_end_date"])
                            writer.writeheader()
                            for i in range(len(dataset)):
                                for group in analysis.GROUPS:
                                    writer.writerow({"window_index":i,"group":group,"input_end_date":dataset.window_end_date(i)})
        return roots,evidence,scalers,cat

    def analyze(self,roots,evidence,scalers,cat):
        def replay(ticker,args,data_root,categories=False):
            return evidence,scalers,cat if categories else {}
        with patch.object(analysis,"replay_windows",side_effect=replay),patch.object(analysis,"load_manifest",return_value={}), \
             patch.object(analysis,"file_hash",return_value="verified_manifest"), \
             patch.object(analysis,"verify_old_uniform",return_value={"status":"passed","max_prediction_abs_diff":0}), \
             contextlib.redirect_stdout(io.StringIO()):
            return analysis.analyze(*roots)

    def test_four_conditions_statistics_partial_legacy_and_primary_validation_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            roots,evidence,scalers,cat=self.fixture(Path(directory))
            tables,selection=self.analyze(roots,evidence,scalers,cat)
            self.assertEqual(len(tables["per_run"]),120)
            self.assertEqual(selection["selected_variants"],["uniform"])
            self.assertEqual(set(selection["validation_scores"]),{"attention","uniform"})
            self.assertFalse(selection["test_used_for_selection"])
            self.assertEqual(sum(r["window_verification"]=="full" for r in tables["per_run"]),60)
            self.assertEqual(sum(r["window_verification"]=="partial" for r in tables["per_run"]),60)
            self.assertEqual(len(tables["old_price_only_reproduction"]),30)
            row=next(r for r in tables["across_seeds"] if r["variant"]=="attention" and r["metric"]=="val_mse")
            self.assertAlmostEqual(row["mean"],.35)
            self.assertAlmostEqual(row["sample_std"],.05)
            with self.assertRaises(FileExistsError):
                self.analyze(roots,evidence,scalers,cat)

    def test_primary_mismatches_block_selection_but_save_reports(self):
        for change in ("categories","targets","config","partial"):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as directory:
                roots,evidence,scalers,cat=self.fixture(Path(directory))
                output=roots[0]/"full/uniform/AAPL/seed7/output"
                if change=="targets":
                    path=output/"test_io.pt"
                    payload=torch.load(path,weights_only=True)
                    payload["ground_truths"]+=1
                    torch.save(payload,path)
                else:
                    path=output/"best_checkpoint.pt"
                    payload=torch.load(path,weights_only=True)
                    if change=="categories":
                        payload["args"]["category_window_evidence"]["val"]["category_sha"]="wrong"
                    elif change=="config":
                        payload["args"]["category_scorer_config"]["hidden_size"]=32
                    else:
                        payload["args"].pop("window_evidence")
                        sp=output/"summary.json"
                        summary=json.loads(sp.read_text());summary.pop("window_evidence");sp.write_text(json.dumps(summary))
                    torch.save(payload,path)
                tables,selection=self.analyze(roots,evidence,scalers,cat)
                self.assertEqual(selection["selection_status"],"blocked_primary_comparability")
                self.assertEqual(selection["selected_variants"],[])
                self.assertTrue(tables["issues"])
                self.assertTrue((roots[0]/"diagnostics/comparison_10/window_verification.csv").is_file())
