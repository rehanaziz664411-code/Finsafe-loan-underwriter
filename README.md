# 🏦 FinSafe AI | Corporate Loan Underwriting Portal

**FinSafe AI** is an automated credit risk assessment engine designed for financial institutions to streamline the loan approval process. By analyzing 4,300+ historical loan benchmarks, the system provides a probabilistic risk score based on applicant financials, asset liquidity, and credit history (CIBIL).

## 📊 Engineering Methodology

> [!TIP]
> **Audit Ready:** This model utilizes **Stratified K-Fold Cross-Validation** to ensure stable performance across small-to-medium datasets (4k+ rows).

### Technical Specifications
| Attribute | Specification |
| :--- | :--- |
| **Model Engine** | Random Forest Classifier (Optimized for Small Data) |
| **Preprocessing** | Robust Label Encoding & Standard Scaling |
| **Risk Factors** | Income-to-Asset Ratio, CIBIL Score, Collateral Coverage |
| **Training Basis** | 4,300 Historical Banking Benchmarks |
| **Deployment** | Streamlit Corporate UI |

## 🚀 Key Features
* **Multi-Asset Evaluation:** Analyzes residential, commercial, luxury, and bank assets to determine collateral strength.
* **Risk Tolerance Slider:** Allows bank managers to adjust the decision threshold based on current economic volatility.
* **Underwriting Transparency:** Provides a system confidence percentage for every approval/rejection.

## 🔧 Installation & Usage
1. **Clone the Repo:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/finsafe-loan-underwriter.git](https://github.com/YOUR_USERNAME/finsafe-loan-underwriter.git)
