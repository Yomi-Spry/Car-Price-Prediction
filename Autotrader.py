import pickle
import streamlit as st
import pandas as pd
from datetime import datetime 

model = pickle.load(open('trained_model.sav', 'rb'))
pipe = pickle.load(open('cleaning_pipe.sav', 'rb'))
auto = pickle.load(open('auto_pipe.sav', 'rb'))

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
    return f"£{prediction[0]}"

def main():
    
    st.title('Auto Trader Predictor App')
    mileage= st.number_input("MILEAGE (NOT EXCEEDING 126000)", value=0.0, max_value=126000.0, step=1.0)
    
    stand_colors = ['Red', 'Yellow', 'Blue','White','Beige','Black','Bronze','Brown','Gold'\
                    'Green','Multicolour','Orange','Purple','Silver']

    standard_colour = st.selectbox("COLOUR", stand_colors + ['Other'])
    if standard_colour == "Other":
        standard_colour = st.text_input("Enter a colour")
    
    make=['SEAT','DS AUTOMOBILES','Dacia','Volkswagen', 'Volvo', 'Kia','SKODA','Land Rover',\
          'Peugeot','BMW','Audi','Renault','Jaguar','Nissan','Mercedes-Benz','MINI','Lexus',\
          'Vauxhall','Citroen','Honda','Suzuki','Mazda','Chrysler','MG','Fiat','Porsche',\
          'Mitsubishi','Hyundai','Toyota','Alfa Romeo','Jeep','Subaru','Bentley','Abarth',\
          'SsangYong','Smart','Maserati','Chevrolet','Aston Martin','Infiniti','Morgan',\
          'Dodge','Tesla','Saab','Lotus','Isuzu','Caterham','Daimler','Cadillac','CUPRA','Hummer',\
          'Alpine','Daihatsu','Pilgrim','Holden']
    standard_make = st.selectbox("MAKE",make)
    
    models= ['Golf','Corsa','C Class','3 Series','1 Series','Qashqai','Polo','Hatch','Astra','A3',\
            'A Class','Juke','500','E Class','A1','Sportage','Yaris','Clio','Range Rover Evoque',\
            'Leon','5 Series','AYGO','Octavia','208','Fabia','Tiguan','Insignia','Captur','Jazz',\
            'Civic','Ibiza','A5','i10','Mokka X','3008','4 Series','2008','Tucson','XC60','Passat',\
            'A4','Auris','Q3','Discovery Sport','XF','CR-V','C1','V40','Range Rover Sport','Countryman',\
            'X1','Micra','Kadjar','C3','Outlander','TT','Picanto','up!','108','Q2','Mazda3','Duster',\
            'Ceed','308','Swift','2 Series','Q5','GLA Class','Zafira Tourer','Mokka','Superb','C-HR',\
            'Mazda2','X-Trail','A4 Avant','XE','XC40','Mazda6','CLA Class','Crossland X','Ateca','Vitara',\
            'T-Roc','A6 Saloon','i20','Scirocco','Megane','B Class','Karoq','X3','i30','F-PACE','GLC Class',\
            'Convertible','X5','4 Series Gran Coupe','Grand C4 Picasso','Arona','CX-5','Note','Leaf','ADAM',\
            'Clubman','S3','Range Rover','Prius','RAV4','500X','XC90','C4 Cactus','Meriva','Grandland X',\
            'Sandero Stepway','E-PACE','Yeti','Corolla', 'DS3','V60','Rio','fortwo','Citigo','Renegade','Ioniq',\
            'DS 3','Touran','2 Series Active Tourer','A6 Avant','Panda','Zoe',
            'Venga','2 Series Gran Tourer','595','Zafira','107','Astra GTC','Niro','C3 Aircross','Q7','S Class',\
             'CT 200h',
             'GLE Class',
             'SLK',
             'T-Cross',
             'CLS',
             'S60',
             'Ignis',
             'Sandero',
             'Kona',
             'Z4',
             'ASX',
             'ix35',
             'Kodiaq',
             'Kamiq',
             'Range Rover Velar',
             'HR-V',
             'C4 Picasso',
             'NX 300h',
             'CX-3',
             'MG ZS',
             'Freelander 2',
             'M Class',
             'Stonic',
             '500C',
             '7 Series',
             'Sharan',
             'M4',
             'Touareg',
             'Discovery 4',
             'IS 300',
             'MX-5',
             'Sorento',
             'Discovery',
             'V90',
             'C3 Picasso',
             'Santa Fe',
             'Tipo',
             'Giulietta',
             'A5 Cabriolet',
             'i3',
             'Compass',
             'SX4 S-Cross',
             'Viva',
             'Avensis',
             'A3 Cabriolet',
             'forfour',
             'A7',
             '207',
             'Berlingo',
             'F-Type',
             'X2',
             'S5',
             '5008',
             'Cayenne',
             'Beetle',
             'Eclipse Cross',
             'M3',
             'XJ',
             'V40 Cross Country',
             'ix20',
             'Alhambra',
             'S90',
             'MG3',
             'Celerio',
             'C5 Aircross',
             'Giulia',
             'Korando',
             'SQ5',
             'Grand Scenic',
             'CC',
             '6 Series',
             'MiTo',
             'Verso',
             'X6',
             'Scala',
             'RX 450h',
             'Soul',
             '508',
             'Twingo',
             'Qashqai+2',
             '500L',
             '308 SW',
             'Macan',
             'Pulsar',
             'SLC',
             '3 Series Gran Turismo',
             'i40',
             'A8',
             'X4',
             'Forester',
             'C4',
             'DS 3 CROSSBACK',
             'Wrangler',
             'Mii',
             'Golf SV',
             'Alto',
             'ProCeed',
             'C30',
             'Prius+',
             'Scenic',
             'XCeed',
             'TTS',
             'Rapid Spaceback',
             'UX 250h',
             'Antara',
             '911',
             'Coupe',
             'V70',
             'Partner Tepee',
             'V90 Cross Country',
             'SL Class',
             'A4 Allroad',
             'Tivoli',
             'Carens',
             'M2',
             'Mirage',
             'Tarraco',
             'V Class',
             'Jimny',
             'Boxster',
             'DS 7 CROSSBACK',
             'Optima',
             'CX-30',
             'Punto',
             'Cayman',
             '508 SW',
             'i800',
             '9-3',
             'S4',
             'Impreza',
             'RS3',
             'Panamera',
             'Shogun Sport',
             'Stelvio',
             'Tiguan Allspace',
             'Combo Life',
             'Rapid',
             'Agila',
             'Arteon',
             '6 Series Gran Coupe',
             'DS4',
             'Continental',
             'X-Type',
             'MX-5 RF',
             'CLK',
             'Model S',
             'GL Class',
             'XV',
             'Q30',
             'Logan MCV',
             'Jetta',
             'Shogun',
             'Captiva',
             'Paceman',
             'S1',
             '595C',
             'Rifter',
             'M5',
             'Ghibli',
             'Grande Punto',
             'Outback',
             'Splash',
             'Grand Cherokee',
             'RS6 Avant',
             'RCZ',
             'Baleno',
             'A6 Allroad',
             'L200',
             'Punto Evo',
             'Accord',
             'MG HS',
             'Roomster',
             'iQ',
             'S40',
             'Caravelle',
             '718 Cayman',
             'GLB Class',
             'V50',
             'Caddy Maxi Life',
             'CLC Class',
             'Defender 90',
             'SX4',
             '124 Spider',
             'Stinger',
             'Turismo',
             'RS5',
             'Rexton',
             'GT86',
             'S4 Avant',
             'SQ7',
             'XC70',
             'Eos',
             'Vito',
             '2 Series Gran Coupe',
             'Cherokee',
             'Mazda5',
             'Land Cruiser',
             'DS5',
             '207 SW',
             'DS 4',
             'Toledo',
             'Koleos',
             'ID.3',
             'V60 Cross Country',
             'Honda E',
             'Levante',
             'Golf Plus',
             '6 Series Gran Turismo',
             '5 Series Gran Turismo',
             'Roadster',
             'Navara',
             'Hilux',
             'IS 250',
             'XK',
             'XKR',
             '718 Boxster',
             'Pixo',
             'R8',
             'Grand Vitara',
             'Defender 110',
             'Grand C4 SpaceTourer',
             'Quattroporte',
             'Spark',
             'Estima',
             'Altea',
             'Doblo',
             'DS 4 CROSSBACK',
             'Altea XL',
             'TT RS',
             'Kangoo',
             '207 CC',
             'Fox',
             'M6',
             'I-PACE',
             '159',
             'RS Q3',
             'RC 300h',
             'GS',
             'DS3 Cabrio',
             'Colt',
             'Logan MCV Stepway',
             'Lancer',
             'C70',
             'Vectra',
             'Corolla Verso',
             '308 CC',
             'CL',
             'Vivaro',
             'Vantage',
             'Discovery 3',
             'GLS Class',
             'A4 Cabriolet',
             'CR-Z',
             'X5 M',
             '407',
             'i8',
             'Aveo',
             'Cruze',
             'Caddy Maxi',
             'Viano',
             '9-5',
             'S80',
             'Sedona',
             'Grand Modus',
             'Q50',
             'Qubo',
             'PROACE Verso',
             'DS 5',
             'Laguna',
             'Insight',
             'Amarok',
             'D-Max',
             'IS 220d',
             'A6 Unspecified',
             'Grand Espace',
             'Levorg',
             'Veloster',
             'C5',
             'Phaeton',
             'Tivoli XLV',
             'DS 3 CABRIO',
             'FR-V',
             'ES 300h',
             'DB9',
             'GS 450h',
             'Camry',
             'VXR8',
             'Ypsilon',
             'RX 400h',
             'Wind',
             'SQ2',
             'Model 3',
             'GT-R',
             '370 Z',
             'GS 300',
             'Pathfinder',
             'Grand Voyager',
             'Legacy',
             'Granturismo',
             'Q8',
             'GranCabrio',
             'Cube',
             'Elgrand',
             'Exeo',
             'Musso',
             'Tigra',
             'SpaceTourer',
             'C4 SpaceTourer',
             'Expert Tepee',
             'Spider',
             'Supra',
             'Orlando',
             'Bipper Tepee',
             'Serena',
             'Trax',
             'MG6',
             'Modus',
             'Brera',
             'RS4 Avant',
             'Urbancruiser',
             'Transporter Sportline',
             '300C',
             'Xsara Picasso',
             'Cascada',
             'WRX STI',
             'Alphard',
             'RS4',
             'RS7',
             'S6 Saloon',
             'S-Type',
             'RC F',
             '307',
             'C-Crosser',
             'Patriot',
             'Nitro',
             'Evora',
             'Bravo',
             'e-NV200',
             '159 Sportwagon',
             'C2',
             'Integra',
             'eNV200 Evalia',
             'Transporter Shuttle',
             'Verso S',
             'Seven',
             'Transporter',
             'CX-7',
             '307 CC',
             '695',
             'Micra C+C',
             'Caddy Life',
             'Camaro',
             '147',
             'BRZ',
             '206',
             'RX-8',
             'Matiz',
             'Nemo Multispace',
             'Previa',
             'RS4 Cabriolet',
             'Scenic Xmod',
             'Kalos',
             'G Class',
             'Alpina B5',
             'Traveller',
             'Vivaro Life',
             'AMG',
             'Dispatch',
             'LS 600h',
             'Trafic',
             'Lacetti',
             'GS 250',
             'Arnage',
             '4C',
             'Q70',
             'Alpina D3 Bi-Turbo',
             'Stepwagon',
             'HiAce',
             'Grandis',
             'i-MiEV',
             'RAM',
             'Flying Spur',
             'X6 M',
             'Ampera',
             'Elise',
             'NX Unspecified',
             'Twizy',
             'GT',
             'Land Cruiser Amazon',
             'Voxy',
             'QX70',
             'Exige',
             'Cygnet',
             'M6 Gran Coupe',
             'Partner',
             'Relay',
             'S8',
             'Campervan',
             'H2',
             'Odyssey',
             'Vellfire',
             'Scudo',
             'Delica',
             'Z4M',
             '350 Z',
             'Crossfire',
             'Freed',
             'EX',
             'Colt Cabriolet',
             'Corvette',
             '350',
             'Sumo',
             'A110',
             '407 SW',
             'Legend',
             'Sienta',
             'Master',
             'HSV',
             'X Class',
             'Terios',
             'Justy',
             'quattro',
             'i',
             'X6M',
             'Bongo',
             'Sirion',
             'Defender 130',
             'California',
             'Elysion',
             'RS6',
             'LS 460',
             'Boxer',
             'iOn',
             'BLS',
             'Kyron',
             'S2000',
             'Terracan',
             'Super Eight',
             'Celsior',
             'Challenger',
             'R Class',
             'Sebring',
             'X5M',
             'Skyline',
             'Expert',
             'FX',
             'IS Unspecified',
             'Q60',
             'S6 Avant',
             'IS 200',
             'TF',
             'Exiga',
             'Kangoo Maxi',
             'Getz',
             'S7',
             'Fluence',
             'XKR-S',
             'Ducato',
             'Defender Unspecified',
             'C-Zero',
             'STS',
             'H3',
             'Fullback',
             'Sprinter',
             'RX 200t',
             '190',
             'M',
             '4007',
             'IS F',
             'Signum']
    standard_model= st.selectbox("MODEL",models)
    
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
        
    
    #min_date = datetime(2006, 1, 1).date()
   # max_date = datetime.today().date()
    #value_d  = datetime(2006, 1, 1).date()
    date_of_advert = datetime.today().date()
    #date_of_advert = st.date_input('DATE OF ADVERT', min_value=min_date, max_value=max_date,value=value_d)
    date_of_advert = date_of_advert.strftime("%Y%m%d")
   
   
    
    price= ''
    
    if st.button('predict price'):
        price= predictors (mileage,standard_colour,standard_make,standard_model,\
                        vehicle_condition,year_of_registration,body_type,crossover_car_and_van, fuel_type,date_of_advert)
    st.success(price)
    

if __name__=='__main__':
    main()
    