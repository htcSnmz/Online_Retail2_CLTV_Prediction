"""
BG-NBD ve Gamma-Gamma ile CLTV Tahmini
CLTV Prediction with BG-NBD and Gamma-Gamma
-------------------------------------------
İş Problemi:
İngiltere merkezli perakende şirketi satış ve pazarlama
faaliyetleri için roadmap belirlemek istemektedir. Şirketin
orta uzun vadeli plan yapabilmesi için var olan müşterilerin
gelecekte şirkete sağlayacakları potansiyel değerin
tahmin edilmesi gerekmektedir.

Veri Seti Hikayesi:
Online Retail II isimli veri seti İngiltere merkezli bir perakende şirketinin 01/12/2009 - 09/12/2011 tarihleri
arasındaki online satış işlemlerini içeriyor. Şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır ve çoğu
müşterisinin toptancı olduğu bilgisi mevcuttur.

Değişkenler:
8 Değişken 541.909 Gözlem 45.6MB
InvoiceNo Fatura Numarası ( Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder )
StockCode Ürün kodu ( Her bir ürün için eşsiz )
Description Ürün ismi
Quantity Ürün adedi ( Faturalardaki ürünlerden kaçar tane satıldığı)
InvoiceDate Fatura tarihi
UnitPrice Fatura fiyatı ( Sterlin )
CustomerID Eşsiz müşteri numarası
Country Ülke ismi
"""

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

df_ = pd.read_excel("CLTV_Prediction/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.info()
df.describe().T
df = df[~df["Invoice"].str.startswith("C", na=False)]
df = df[df["Quantity"] > 1]
df = df[df["Price"] > 0]
df["TotalPrice"] = df["Quantity"] * df["Price"]

def outlier_threshold(dataframe, variable):
    q1 = dataframe[variable].quantile(0.01)
    q3 = dataframe[variable].quantile(0.99)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_threshold(dataframe, variable)
    dataframe.loc[dataframe[variable] > up_limit, variable] = round(up_limit, 0)
    dataframe.loc[dataframe[variable] < low_limit, variable] = round(low_limit,0)

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

# Görev 1: BG-NBD ve Gamma-Gamma Modellerini Kurarak 6 Aylık CLTV Tahmini Yapılması
# Adım 1: 2010-2011 yıllarındaki veriyi kullanarak müşteriler için 6 aylık CLTV tahmini yapınız.
# Adım 2: Elde ettiğiniz sonuçları yorumlayıp, değerlendiriniz.
analysis_date = df["InvoiceDate"].max() + dt.timedelta(days=2)
cltv_df = df.groupby("Customer ID").agg({
    "InvoiceDate": [lambda date: (date.max() - date.min()).days, lambda date: (analysis_date - date.min()).days],
    "Invoice": lambda invoice: invoice.nunique(),
    "TotalPrice": "sum"})
cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["recency", "T", "frequency", "monetary"]
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7
cltv_df = cltv_df[cltv_df["frequency"] > 1]
cltv_df["avg_monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])
cltv_df["exp_sales_6_month"] = bgf.predict(24, cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df["frequency"], cltv_df["avg_monetary"])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"], cltv_df["avg_monetary"])
cltv_df["cltv"] = ggf.customer_lifetime_value(bgf,
                            cltv_df["frequency"],
                            cltv_df["recency"],
                            cltv_df["T"],
                            cltv_df["avg_monetary"],
                            time=6,
                            discount_rate=0.01,
                            freq="W")

# Görev 2: Farklı Zaman Periyotlarından Oluşan CLTV Analizi
# Adım 1: 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.
# Adım 2: 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz.
uk_customer_ids = (df[df["Country"] == "United Kingdom"]["Customer ID"]).unique()
uk_cltv_df = cltv_df[cltv_df.index.isin(uk_customer_ids)]
bgf.fit(uk_cltv_df["frequency"], uk_cltv_df["recency"], uk_cltv_df["T"])
ggf.fit(uk_cltv_df["frequency"], uk_cltv_df["avg_monetary"])
uk_cltv_df["cltv_1_month"] = ggf.customer_lifetime_value(bgf,
                                                         uk_cltv_df["frequency"],
                                                         uk_cltv_df["recency"],
                                                         uk_cltv_df["T"],
                                                         uk_cltv_df["avg_monetary"],
                                                         time=1,
                                                         discount_rate=0.01,
                                                         freq="W")
uk_cltv_df["cltv_12_month"] = ggf.customer_lifetime_value(bgf,
                                                         uk_cltv_df["frequency"],
                                                         uk_cltv_df["recency"],
                                                         uk_cltv_df["T"],
                                                         uk_cltv_df["avg_monetary"],
                                                         time=12,
                                                         discount_rate=0.01,
                                                         freq="W")
uk_cltv_df.sort_values("cltv_1_month", ascending=False).head(10)
uk_cltv_df.sort_values("cltv_12_month", ascending=False).head(10)

# Görev 3: Segmentasyon ve Aksiyon Önerileri
# Adım 1: 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve
# grup isimlerini veri setine ekleyiniz.
# Adım 2: 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.
uk_cltv_df["cltv_6_month"] = ggf.customer_lifetime_value(bgf,
                                                         uk_cltv_df["frequency"],
                                                         uk_cltv_df["recency"],
                                                         uk_cltv_df["T"],
                                                         uk_cltv_df["avg_monetary"],
                                                         time=6,
                                                         discount_rate=0.01,
                                                         freq="W")
uk_cltv_df["segment"] = pd.qcut(uk_cltv_df["cltv_6_month"], 4, ["D", "B", "C", "A"])
uk_cltv_df.groupby("segment").agg(["mean", "sum", "count"])
