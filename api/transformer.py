import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CreditFeatureAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_out = pd.DataFrame(X).copy()
        
        # 1. Total Bill Amount across 6 months
        bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
        X_out['Total_Bill_Amount'] = X_out[bill_cols].sum(axis=1)
        
        # 2. Total Payment Amount across 6 months
        pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
        X_out['Total_Payment_Amount'] = X_out[pay_amt_cols].sum(axis=1)
        
        # 3. Credit Utilization Ratio
        X_out['Credit_Utilization'] = X_out['Total_Bill_Amount'] / (X_out['LIMIT_BAL'] + 1e-5)
        
        # 4. Max Delay Status
        pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
        X_out['Max_Payment_Delay'] = X_out[pay_status_cols].max(axis=1)
        
        return X_out