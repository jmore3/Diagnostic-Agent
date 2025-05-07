import numpy as np
import pandas as pd
import pytest

from src.diagnostic_agent import build_joint_probability, inference_by_enumeration

@pytest.fixture
def synthetic_df():
    # Construct a tiny DataFrame with the same 8 features + Condition
    data = {
        "Fever":             ["Yes",  "No",   "Yes", "No"],
        "Cough":             ["No",   "Yes",  "No",  "Yes"],
        "Fatigue":           ["Yes",  "Yes",  "No",  "No"],
        "Difficulty Breathing": ["No","No",   "Yes", "Yes"],
        "Age":               [25,     35,     45,    55],
        "Gender":            ["Male", "Female","Male","Female"],
        "Blood Pressure":    ["Low",  "Normal","High","Low"],
        "Cholesterol Level": ["High", "High",  "Low", "Normal"],
        "Disease":           ["A",    "B",     "A",   "C"]
    }
    df = pd.DataFrame(data)
    # Encode exactly as in your main.py
    df["Symptom1"] = df["Fever"].map({"No":0,"Yes":1})
    df["Symptom2"] = df["Cough"].map({"No":0,"Yes":1})
    df["Symptom3"] = df["Fatigue"].map({"No":0,"Yes":1})
    df["Symptom4"] = df["Difficulty Breathing"].map({"No":0,"Yes":1})
    df["Age"] = df["Age"].astype(int)
    df["AgeBin"] = pd.cut(df["Age"], bins=[0,20,30,40,50,60,999],
                          labels=[0,1,2,3,4,5]).astype(int)
    df["GenderCode"] = df["Gender"].map({"Male":0,"Female":1})
    df["BPCode"]   = pd.Categorical(df["Blood Pressure"],
                                    categories=["Low","Normal","High"]).codes
    df["CholCode"] = pd.Categorical(df["Cholesterol Level"],
                                    categories=["Low","Normal","High"]).codes
    df["Condition"] = pd.Categorical(df["Disease"]).codes
    return df

def test_build_joint_and_marginals(synthetic_df):
    arr = synthetic_df[
        ["Symptom1","Symptom2","Symptom3","Symptom4","AgeBin",
         "GenderCode","BPCode","CholCode","Condition"]
    ].values
    joint = build_joint_probability(arr, alpha=0)
    # Joint count for the first row config should be exactly 1 / total
    total = arr.shape[0]
    idx = tuple(arr[0].astype(int))
    assert pytest.approx(joint[idx]) == 1/total

def test_inference_enumeration_simple(synthetic_df):
    arr = synthetic_df[
        ["Symptom1","Symptom2","Symptom3","Symptom4","AgeBin",
         "GenderCode","BPCode","CholCode","Condition"]
    ].values
    joint = build_joint_probability(arr, alpha=0)
    # Query: Fever=Yes and Cough=No and everything else hidden, query Condition
    qv = [1, 0, -1, -1, -1, -1, -1, -1, -2]
    post = inference_by_enumeration(joint, qv)
    # Since in synthetic_df only Disease A appears with (1,0) once and B appears 0
    # post[Condition(A)] should be 1.0
    cond_A = pd.Categorical(synthetic_df["Disease"]).categories.get_loc("A")
    assert post[cond_A] == pytest.approx(1.0)
