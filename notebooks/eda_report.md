
# EDA Report - Breast Cancer Dataset

## 📊 Dataset Overview
- **Shape**: (569, 33)
- **Target Distribution**: {'B': 357, 'M': 212}
- **Numerical Features**: 32
- **Missing Values**: 569

## 🎯 Target Analysis
- **Benigno (B)**: 357 (62.7%)
- **Maligno (M)**: 212 (37.3%)

## 🏆 Top 5 Features per Correlazione con Target
- **concave points_worst**: 0.794
- **perimeter_worst**: 0.783
- **concave points_mean**: 0.777
- **radius_worst**: 0.776
- **perimeter_mean**: 0.743

## 🔍 Outliers Summary
- **radius_mean**: 14 outliers (2.5%)
- **texture_mean**: 7 outliers (1.2%)
- **perimeter_mean**: 13 outliers (2.3%)
- **area_mean**: 25 outliers (4.4%)
- **smoothness_mean**: 6 outliers (1.1%)
- **compactness_mean**: 16 outliers (2.8%)
- **concavity_mean**: 18 outliers (3.2%)

## ✅ Conclusioni
1. Dataset bilanciato con leggera prevalenza di casi benigni
2. Nessun missing value presente
3. Features fortemente correlate con il target identificate
4. Presenza di outliers in alcune features (normale per dati medici)
5. Dataset pronto per il preprocessing e modeling

## 📈 Prossimi Passi
1. Preprocessing delle features
2. Feature selection basata su correlazioni
3. Train/test split
4. Model development
