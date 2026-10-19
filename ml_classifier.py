"""Use a trained model as the gate in front of the ledger builder.

    from ml_classifier import load_gate
    from sms_parser import extract_transactions
    rows = extract_transactions(records, label_fn=load_gate("models/hybrid.joblib"))
"""
import joblib


def load_gate(path="models/hybrid.joblib"):
    model = joblib.load(path)
    return lambda body, sender=None: model.predict([body], [sender])[0]
