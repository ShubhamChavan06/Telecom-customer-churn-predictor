import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
from time import sleep
import random
from sklearn.ensemble import RandomForestClassifier
st.markdown("<div style = 'background-color:orange;padding:10px;'" " style='color:brown;'""<h1>Telecom Churn Customer Sreamlit App<h1>"
                "</div>",unsafe_allow_html=True)
# st.markdown("<h1 style='color:brown;'>Telecom Churn Customer<h1>",unsafe_allow_html=True)
#st.markdown("<div style = 'background-color:red;padding:10px;'" "<h2> This app predict the telecom customer is churned or not,by using Streamlit app</h2>"
            #"</div>",unsafe_allow_html=True)
#st.write(st.write("""
# Churn Prediction App
#This app predicts the telecom customer churn
#        """)
#)

#st.write("""
# Churn Prediction App
#This app predicts the telecom customer churn
#        """)

st.sidebar.header("User Input Features")
st.sidebar.markdown(""" Example csv file""")


# collect the user input features into df
uploded_file = st.sidebar.file_uploader("upload your csv file",type=["csv"])

if uploded_file is not None:
    input_df = pd.read_csv(uploded_file,index_col=0)
else:
    def user_input_features():
        state =st.sidebar.selectbox("Select State",('KS', 'OH', 'NJ', 'OK', 'AL', 'MA', 'MO', 'LA', 'WV', 'IN', 'RI',
       'MT', 'IA', 'NY', 'ID', 'VT', 'VA', 'TX', 'FL', 'CO', 'AZ', 'SC','NE', 'WY', 'HI', 'IL', 'NH', 'GA', 'AK', 'MD', 'AR', 'WI', 'OR',
       'MI', 'UT', 'CA', 'MN', 'SD', 'NC', 'WA', 'NM', 'NV', 'DC', 'KY','ME', 'MS', 'DE', 'TN', 'PA', 'CT', 'ND'))
        area_code = st.sidebar.selectbox("Please Enter Area code of your city",[408,415,510])
        account_length = st.sidebar.slider("How long the account has been active.",min_value=1,max_value=500,value=2)
        voice_plan = st.sidebar.selectbox("Please select customer taken a voice plan ",["no","yes"])
        voice_messages = st.sidebar.slider("Enter no of customers daily voice messages",min_value=0,max_value=50) 
        intl_plan = st.sidebar.selectbox("Does customer activated a International Plan",["no","yes"]) 
        intl_mins = st.sidebar.number_input("Enter total no of international calling minutes")
        intl_calls = st.sidebar.slider("Enter total no of international calls",min_value=0,max_value=50)
        intl_charge = st.sidebar.slider("Enter total international charge",min_value=1,max_value=500)
        day_mins = st.sidebar.slider("No of minutes customer used service during the day",min_value=1,max_value=300)
        day_calls = st.sidebar.slider("Enter total number of calls during the day",min_value=1,max_value=165)
        day_charge = st.sidebar.slider("Enter total charge during the day",min_value=1,max_value=50)
        eve_mins = st.sidebar.slider("minutes customer used service during the evening",min_value=1,max_value=50)
        eve_calls = st.sidebar.slider("Enter total number of calls during the evening",min_value=0,max_value=200)
        eve_charge = st.sidebar.number_input("Enter total charge during the evening")
        night_mins = st.sidebar.slider("minutes customer used service during the night",min_value=0,max_value=500)
        night_calls = st.sidebar.slider("Enter total number of calls during the night",min_value=0,max_value=200)
        night_charge = st.sidebar.number_input("Enter total charge during the night")
        customer_calls = st.sidebar.number_input("Enter total number of calls to customer service")

        data = {'state' : state,
               'area_code' :area_code,
               'account_length' : account_length,
               'voice_plan' : voice_plan,
               'voice_messages' : voice_messages,
               'intl_plan' : intl_plan,
               'intl_mins' : intl_mins,
               'intl_calls' : intl_calls,
               'intl_charge' : intl_charge,
               'day_mins' : day_mins,
               'day_calls' : day_calls,
               'day_charge' : day_charge,
               'eve_mins' : eve_mins,
               'eve_calls' : eve_calls,
               'eve_charge' : eve_charge,
               'night_mins' : night_mins,
               'night_calls' : night_calls,
               'night_charge' : night_charge,
               'customer_calls' : customer_calls
               }
        features = pd.DataFrame(data,index=[0])
        return features

    input_df = user_input_features()

# combine user input features with entire churn data
# This will be useful for encoding phase
churn_raw = pd.read_csv('cleaned_churn.csv',index_col=0)
churn = churn_raw.drop(columns=['churn'])
df = pd.concat([input_df,churn],axis=0)

# Encoding of ordinal features
encode = ['state','voice_plan','intl_plan']
for col in encode:
    dummy = pd.get_dummies(df[col],prefix=col)
    df = pd.concat([df,dummy],axis=1)
    del df[col]
df = df[:1]
# Display the user input features
st.markdown('#### User Input Features')

if uploded_file is not None:
    st.write(df)
else:
    st.write('Awaiting for csv file to upload,currently using example input parameters(shown below)')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('churn_clf.pkl','rb'))

# Apply model to make predictions 
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


with st.spinner("Waiting...!!!"):
    sleep(5)
st.success("Finished")

st.subheader('Prediction')
customer_churn = np.array(['no','yes'])
st.write(customer_churn[prediction])
#churn_threshold = 0.5


if (prediction >= 0.5).any():
    st.success("Yes, the customer is likely to churn.")
else:
    st.success("No, the customer is not likely to churn.")
    
st.subheader("Prediction Probability")
st.write(prediction_proba)
