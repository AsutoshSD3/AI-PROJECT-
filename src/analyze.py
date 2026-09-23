"""Produce repeatable EDA and explainability artifacts from the generated data and saved model."""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import joblib
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import DATA, MODELS, RESULTS, NUMERIC

def main():
    df=pd.read_csv(DATA/"raw"/"pac_libero_lite.csv"); RESULTS.mkdir(exist_ok=True)
    numeric=df[NUMERIC]
    summary={"shape":[int(df.shape[0]),int(df.shape[1])],"dtypes":{k:str(v) for k,v in df.dtypes.items()},"missing_by_column":df.isna().sum().astype(int).to_dict(),"duplicate_records":int(df.duplicated().sum()),"role_distribution":df.role.value_counts().to_dict(),"action_distribution":df.action_label.value_counts().to_dict(),"numeric_summary":numeric.describe().round(4).to_dict(),"correlation":numeric.corr().round(4).to_dict()}
    (RESULTS/"eda_summary.json").write_text(json.dumps(summary,indent=2))
    fig,axs=plt.subplots(1,3,figsize=(13,3.6)); df.role.value_counts().reindex(["referential","incidental","conflicting"]).plot.bar(ax=axs[0],color=["#51a86d","#4f8dc9","#e66c50"]); axs[0].set_title("Provenance-class distribution"); axs[0].set_xlabel(""); axs[0].set_ylabel("samples")
    df.action_label.value_counts().reindex(["left","center","right"]).plot.bar(ax=axs[1],color="#355c7d"); axs[1].set_title("Expert action distribution"); axs[1].set_xlabel(""); axs[1].set_ylabel("samples")
    im=axs[2].imshow(numeric.corr(),vmin=-1,vmax=1,cmap="coolwarm"); axs[2].set_title("Numeric-feature correlation"); axs[2].set_xticks(range(len(NUMERIC)),NUMERIC,rotation=70,fontsize=7); axs[2].set_yticks(range(len(NUMERIC)),NUMERIC,fontsize=7); fig.colorbar(im,ax=axs[2],fraction=.046)
    fig.tight_layout(); fig.savefig(RESULTS/"eda_overview.png",dpi=180); plt.close(fig)
    artifact=joblib.load(MODELS/"pac_bc_pipeline.joblib"); pipe=artifact["action_model"]; model=pipe.named_steps["model"]; names=pipe.named_steps["prep"].get_feature_names_out()
    if hasattr(model,"coef_"):
        importance=pd.DataFrame({"feature":names,"importance":abs(model.coef_).mean(axis=0)}).sort_values("importance",ascending=False).head(20)
    else:
        importance=pd.DataFrame({"feature":names,"importance":model.feature_importances_}).sort_values("importance",ascending=False).head(20)
    importance.to_csv(RESULTS/"action_feature_importance.csv",index=False)
    print(json.dumps({"samples":len(df),"eda_plot":str(RESULTS/"eda_overview.png"),"top_feature":importance.iloc[0].to_dict()},indent=2))
if __name__=="__main__": main()
