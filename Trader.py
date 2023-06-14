import pickle
import streamlit as st
import pandas as pd
from datetime import datetime 

model = pickle.load(open('trained_model.sav', 'rb'))
pipe = pickle.load(open('cleaning_pipe.sav', 'rb'))
auto = pickle.load(open('auto_pipe.sav', 'rb'))
df= pd.read_csv("sample.csv")

def predictors (mileage,standard_colour,standard_make,standard_model,\
                vehicle_condition,year_of_registration,body_type,crossover_car_and_van,\
                   fuel_type,date_of_advert):
    
    data = pd.DataFrame({'mileage': [mileage], 'standard_colour': [standard_colour],
                         'standard_make': [standard_make],\
                         'standard_model': [standard_model],'vehicle_condition': [vehicle_condition],\
                        'year_of_registration': [year_of_registration],'body_type':[body_type],\
                        'crossover_car_and_van':[crossover_car_and_van],\
                        'fuel_type':[fuel_type],
                        'date_of_advert':[date_of_advert]})
    def mileage_group(mileage):
        if mileage < 100:
            cat = 'Fresh'
        elif mileage >= 100 and mileage < 1000:
            cat = 'Fairly fresh'
        elif mileage >= 1000 and mileage < 10000:
            cat = 'Good'
        elif mileage >= 10000 and mileage < 50000:
            cat = 'Worked'
        elif mileage >= 50000 and mileage < 100000:
            cat = 'Aged'
        elif mileage >= 100000 and mileage < 500000:
            cat = 'Old'
        else:
            cat = 'Wrecked'
        return cat
        
    data['engine_condition']=data['mileage'].apply(mileage_group)
    data=pipe.transform(data)
    data=auto.transform(data)
    prediction=model.predict(data)
    return f"£{round(prediction[0], 2)}"




def main():
    
    st.title('Auto Trader Predictor App')
    mileage= st.number_input("MILEAGE (NOT EXCEEDING 126000)", value=0.0, max_value=126000.0, step=1.0)
    
    stand_colors = ['Red', 'Yellow', 'Blue','White','Beige','Black','Bronze','Brown','Gold'\
                    'Green','Multicolour','Orange','Purple','Silver']

    standard_colour = st.selectbox("COLOUR", stand_colors + ['Other'])
    if standard_colour == "Other":
        standard_colour = st.text_input("Enter a colour")

    make= df['standard_make'].drop_duplicates().values.tolist()
    standard_make = st.selectbox("MAKE",make)

    s_models= df['standard_model'].drop_duplicates().values.tolist()
    standard_model= st.selectbox("MODEL",s_models)

    vehicle_condition = st.radio("VEHICLE CONDITION", ["NEW", "USED"])
    

    year_of_registration = st.slider("YEAR OF REGISTRATION", min_value=2006, max_value=2021, value=2021)
   
    
    btype = ['Convertible', 'Coupe', 'Estate','Hatchback','MPV','Pickup','SUV','Saloon']
    body_type = st.selectbox("BODY TYPE", btype + ['Other'])
    if body_type == "Other":
        body_type = st.text_input("BODY TYPE")
    
    crossover_car_and_van = st.radio("IS CAR A CROSSOVER CAR AND VAN?", ["True", "False"])
    
    
    ftype = ['Diesel', 'Diesel Hybrid', 'Electric','Petrol','Petrol Hybrid','Plug-in Hybrid']
    fuel_type = st.selectbox("FUEL TYPE", ftype + ['Other'])
    if fuel_type == "Other":
        fuel_type = st.text_input("FUEL TYPE")
        
    date_of_advert = datetime.today().date()
    date_of_advert = date_of_advert.strftime("%Y%m%d")
   
   
    
    price= ''
    
    if st.button('predict price'):
        price= predictors (mileage,standard_colour,standard_make,standard_model,\
                        vehicle_condition,year_of_registration,body_type,crossover_car_and_van, fuel_type,date_of_advert)
    st.success(price)
    

if __name__=='__main__':
    main()