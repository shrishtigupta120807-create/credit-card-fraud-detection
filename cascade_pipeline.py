
class FraudCascade:
    def __init__(self, tier1_model, tier2_model, low_threshold, high_threshold):
        self.tier1 = tier1_model
        self.tier2 = tier2_model
        self.low = low_threshold
        self.high = high_threshold

    def predict(self, transaction):
        score = self.tier1.predict_proba(transaction.values.reshape(1, -1))[0][1]
        if self.low <= score <= self.high:
            return self.tier2.predict_proba(transaction.values.reshape(1, -1))[0][1], "escalated"
        return score, "tier1_only"
