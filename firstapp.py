import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

st.set_page_config(
    page_title="Sharky's Credit Default App",
    page_icon="🤑",
    layout="wide"
)

@st.cache_data
def load_data():
    data = pd.read_csv("prosper_data_app_dev.csv")
    return(data.dropna())

@st.cache_resource
def load_model():
    filename = "finalized_default_model.sav"
    loaded_model = pickle.load(open(filename, 'rb'))        
    return loaded_model


data = load_data()
model = load_model()

#### Define Header of app
#########################################################
st.title("Sharky's Credit Default App")
st.markdown("🤑 This application is a Streamlit dashboard that can be used to *analyze* and **predict** credit default 🤑💥💰")


#### Definition of Section 1 for exploring data
#########################################################

st.header("Customer Explorer")


row1_col1, row1_col2, row1_col3 = st.columns([1,1,1])

rate = row1_col1.slider("Interst the customer has to pay",
                 data["borrower_rate"].min(), 
                 data["borrower_rate"].max(), (0.05, 0.15) )
row1_col1.write(rate)

income = row1_col2.slider("Monthly Income of Customers",
                 data["monthly_income"].min(),
                  data["monthly_income"].max(),
                   (2000.0,30000.0))
row1_col2.write(income)


mask = ~data.columns.isin(["loan_default","employment_status","borrower_rate"])
names = data.loc[:,mask].columns
variable = row1_col3.selectbox("Select Variable to compare", names)
row1_col3.write(variable)

# creatin filtered data set

filtered_data = data.loc[(data["borrower_rate"] >= rate[0]) & 
                         (data["borrower_rate"] <= rate[1]) & 
                         (data["monthly_income"] >= income[0]) & 
                         (data["monthly_income"] <= income[1]),:]

if st.checkbox("Show filtered data", False):
    st.write(filtered_data)


# matplotlib barchart 

barplotdata = filtered_data[["loan_default", variable]].groupby("loan_default").mean()

fig1, ax = plt.subplots(figsize=(8,3.7))
ax.bar(barplotdata.index.astype(str), barplotdata[variable], color="#fc8d62")
ax.set_ylabel(variable)

row2_col1, row2_col2 = st.columns([1,1]) 

row2_col1.subheader("Compare Customer Groups")
row2_col1.pyplot(fig1)


fig2 = sns.lmplot(data=filtered_data, x =variable, y="borrower_rate", order=2, height=4, aspect=1/1, col="loan_default", palette="Set2")
row2_col2.subheader("Borrower rate correlations")
row2_col2.pyplot(fig2, use_container_width=True)



#### Definition of Section 2 for model prediction
#########################################################
st.header("Predicting Cusomter Default")

uploaded_data = st.file_uploader("Choose a file with Customer Data for predicting customer default")

#Getting the data from the uploaded file
if uploaded_data is not None:
    new_customers = pd.read_csv(uploaded_data)
    new_customers = pd.get_dummies(new_customers, drop_first=True)
    new_customers["predicted_default"] = model.predict(new_customers)

    st.download_button(label="Download Predicted Customers",
                   data = new_customers.to_csv().encode('utf-8'),
                   file_name = "predicted_customers.csv")