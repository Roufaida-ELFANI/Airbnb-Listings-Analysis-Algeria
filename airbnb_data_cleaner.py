import pandas as pd
import re
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import itertools
from skopt import BayesSearchCV
from datetime import datetime
from dateutil.relativedelta import relativedelta

class AirbnbDataCleaner:
    def __init__(self, df):
        self.df = df

    def clean_reviews_rating(self):
        self.df['n_reviews'] = (self.df['n_reviews']
                                .str.replace("reviews", "", regex=False)
                                .str.strip()
                                .replace("No", "")
                                .astype(str))
        self.df['n_reviews'] = pd.to_numeric(self.df['n_reviews'], errors='coerce').fillna(0).astype(int)

        self.df['Overall_rating'] = (self.df['Overall_rating']
                                     .str.replace("No reviews yet", "", regex=False)  
                                     .str.replace("New", "", regex=False)           
                                     .str.strip())                                  
        
        self.df['Overall_rating'] = pd.to_numeric(self.df['Overall_rating'], errors="coerce")

        return self.df

    def split_home_pieces(self):
        def extract_home_features(text):
            guests = bedrooms = beds = baths = None
            if pd.notna(text):
                for segment in str(text).split('****'):
                    segment = segment.strip().lower()
                    if 'guest' in segment:
                        guests = int(re.search(r'(\d+)', segment).group(1)) if re.search(r'(\d+)', segment) else None
                    elif 'bedroom' in segment:
                        bedrooms = int(re.search(r'(\d+)', segment).group(1)) if re.search(r'(\d+)', segment) else None
                    elif 'bed' in segment and 'bedroom' not in segment:
                        beds = int(re.search(r'(\d+)', segment).group(1)) if re.search(r'(\d+)', segment) else None
                    elif any(keyword in segment for keyword in ['bath', 'shared bathroom', 'private attached bathroom']):
                        baths = int(re.search(r'(\d+)', segment).group(1)) if re.search(r'(\d+)', segment) else None
            return guests, bedrooms, beds, baths

        self.df[['n_guest', 'n_bedroom', 'n_bed', 'n_bath']] = self.df['Home_pieces'].apply(
            lambda x: pd.Series(extract_home_features(x))
        )
        return self.df

    def split_address(self):
        split_df = self.df['Address'].str.split(',', expand=True).reindex(columns=[0, 1, 2], fill_value='Unknown')
        split_df.columns = ['City', 'Province', 'Country']
        split_df = split_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        self.df[['City', 'Province', 'Country']] = split_df
        return self.df

    def clean_country(self):
        self.df = self.df[self.df["Country"] != "Spain"]
        self.df.loc[self.df["Country"] == "Khareza", "Country"] = "Algeria"
        self.df.loc[~self.df['Address'].isna() & self.df['Country'].isna(), 'Country'] = "Algeria"
        self.df['Country'] = self.df['Country'].str.lower().str.replace("algérie", "Algeria")
        return self.df

    def standardize_city(self, column):
        city_dict = {"tipasa": "Tipaza",
        "alger": "Algiers",
        "béjaïa": "Bejaia",
        "bejaya": "Bejaia",
        "بسكرة": "Biskra",
        "وهران": "Oran",
        "médéa": "Medea",
        "boumerdès": "Boumerdes",
        "béchar": "Bechar",
        "naâma": "Naama",
        "sétif": "Setif",
        "الجلفة": "Djelfa",
        "tébessa": "Tebessa",
        "عين الترك": "Aïn El Turk",
        'el kala': 'El-Kala',
        'sidi bel abbès': 'Sidi-Bel-Abbès',
        'sidi lakhdar': 'Sidi-Lakhdar',
        'aïn témouchent': 'Ain Temouchent',
        'algiers [el djazaïr]': 'algiers',
        'koléa': 'Kolea',
        'aïn temouchent': 'Ain Temouchent',
        'aïn témouchent': 'Ain Temouchent',
        'aïn defla': 'Ain Defla',
        'bouïra': 'Bouira',
        'chaïba': 'Chaiba',
        'aïn turk': 'Aïn El Turk',
        'ain el turck': 'Aïn El Turk',
        'ain el turk' : 'Aïn El Turk',
        'ville': '',
        'oran centre': 'Oran',
        'ain tagourait': 'Aïn Tagourait',
        'khmisti ville': 'Khemisti',
        'bou ismaïl': 'Bou Ismail',
        'chetaïbi': 'Chetaibi',
        'béni saf': 'Beni Saf',
        'bouzedjar': 'Bou Zadjar',
        'ouled rahmoun': 'Ouled Rahmoune',
        'mezghrane': 'Mazagran',
        'mezagheran': 'Mazagran',
        'kharouba plag': 'Kharouba',
        'benabdelmalek ramdane': 'Ben Abdelmalek Ramdan',
        'tahir': 'Taher',
        'douéra': 'Douera',
        'dar el beida': 'Dar El Beïda',
        'aïn taya': 'Ain Taya',
        'souk el ténine': 'Souk El Tenine',
        'el kseur': 'El-Kseur',
        'delles': 'Dellys',
        'el mersa': 'El Marsa',
        'tamentfoust /el Marsa': 'El Marsa',
        'béni oulbane': 'Beni Ouelbane',
        'bou saada': 'Bou Saâda',
        'bordj el bahrj': 'Bordj El Bahri',
        'beni oulbane' : 'Beni Ouelbane',
        'bouismail': 'Bou Ismail',
        'الخروب': 'El Khroub',
        'le khroub': 'El Khroub',
        'بئر الجير': 'Bir El Djir',
        'el ansser': 'El Ançor',
        'el bayadh': 'El-Bayadh',
        'el tarf': 'El Taref',
        "h'raoua": 'Heuraoua'}

        self.df[column] = self.df[column].str.strip().str.lower().replace(city_dict, regex=False)
        self.df[column] = self.df[column].apply(lambda x: x.title() if isinstance(x, str) else x)
        return self.df

    def clean_province(self):
        self.df['Province'] = (self.df['Province']
                          .str.strip()
                          .str.lower()
                          .str.replace(r"wilaya de|wilaya|wilaya d'|province|ولاية|d'", "", regex=True)
                          .str.strip())
        city_to_province = {
            "annaba": "Annaba",
            "bordj el kiffan": "Algiers",
            "ouled rahmoun constantine": "Constantine",
            "tlemcen": "Tlemcen",
            "el tarf": "El Taref",
            "constantine": "Constantine",
            "béjaïa": "Béjaïa",
            "boumerdes": "Boumerdes",
            "oran": "Oran",
            "bejaia": "Béjaïa",
            "misserghin": "Oran",
            "tipaza": "Tipaza",
            "ghardaia": "Ghardaia",
            "jijel": "Jijel",
            "aïn témouchent": "Aïn Témouchent",
            "batna": "Batna",
            "tizi ouzou": "Tizi Ouzou",
            "tigzirt": "Tizi Ouzou",
            "mostaganem": "Mostaganem",
            "ain taya": "Algiers",
            "algiers": "Algiers",
            "borj el kiffan": "Algiers"
        }
    
        dict =  {
        'Heuraoua': 'Algiers',
        'Anar Amelal': 'Tizi Ouzou',
        'قسنطينة': "Constantine",
        "Bordj Bou Arréridj": "Bordj Bou Arreridj",
        "Bejaya": "Bejaia",
        "Heuraoua": "Algiers",
        "Commune De Toudja  Bejaia": "Bejaia"}
        
        for city, province in city_to_province.items():
            self.df.loc[self.df['Province'].isna() & (self.df['City'].str.lower().str.strip() == city), 'Province'] = province
    
        for key, item in dict.items():
            self.df.loc[self.df['Province'].str.lower()==key.lower(), 'Province'] = item
        
        self.df.loc[self.df['Province'].str.lower().isin(['algeria', 'algérie', 'algerie']), 'Province'] = None
    
        self.df = self.standardize_city('Province')
        
        return self.df
    
    # Function to clean city information
    def clean_city(self):
        self.df['City'] = (self.df['City']
                      .str.strip()
                      .str.lower()
                      .str.replace(r"wilaya de|wilaya|wilaya d'|province|ولاية|d'|commune", "", regex=True)
                      .str.strip())
        
        self.df.loc[~self.df['Address'].isna() & self.df['Province'].isna(), 'City'] = None
    
        self.df.loc[self.df['City'].str.lower().str.strip()=="algeria", "City"] = None
    
        self.df.loc[self.df['City'].str.lower().str.strip()=='braidia', "City"] = "Heuraoua"
    
        self.df.loc[self.df['Title'].str.lower().str.strip()=="family apartment", "City"] = "Anar Amelal"
        
        self.df = self.standardize_city('City')
    
        #self.df = self.df.dropna(subset=['City'])
        
        return self.df

    def clean_price(self):
        self.df['Price'] = (self.df['Price']
                            .str.replace('€', '')
                            .str.replace(',', '')
                            .astype(float))
        return self.df

    def split_response_rate(self):
                
        self.df['Response_rate'] = self.df['Response_rate'].fillna('').astype(str)
        split_parts = self.df['Response_rate'].str.split(r'\*\*\*\*', expand=True)
        self.df['response_rate'] = split_parts[0].str.strip()
        self.df['response_time'] = split_parts[1].str.strip() if split_parts.shape[1] > 1 else None
        self.df['response_time'] = self.df['response_time'].replace({'': None, 'nan': None})
        self.df['response_rate'] = self.df['response_rate'].replace({'': None, 'nan': None})
        self.df['response_time'] = self.df['response_time'].str.extract(r'Responds\s+(.*)')
        self.df['response_rate'] = self.df['response_rate'].str.split(':').str[1].str.replace("%", "").str.strip().astype(float)
        return self.df

    def clean_experience(self):
        def df_split(text):
            superhost = host_experience = 0
            if pd.notna(text):
                text = text.strip().lower()
                match = re.findall(r'(\d+)\s*(year|month)', text)
                if match:
                    superhost = 1 if 'superhost' in text else 0
                    host_experience = ' '.join([f"{num} {unit}" for num, unit in match])
                else:
                    host_experience = '1 month'
            return superhost, host_experience

        def standardize_time_experience(text):
            if isinstance(text, str):
                if "month" in text.lower():
                    num_months = ''.join(filter(str.isdigit, text))  
                    if num_months.isdigit():
                        months = int(num_months)
                        return round(months / 12, 1) 

                elif text.split()[0].isdigit():
                    return int(text.split()[0])  # Return the numeric part (e.g., "2 years" -> 2)
        
            return text

        self.df[['superhost', 'host_experience']] = self.df['Experience'].apply(lambda x: pd.Series(df_split(x)))
        
        self.df['host_experience'] = self.df['host_experience'].apply(standardize_time_experience)
        
        return self.df

    def split_additional_fees_column(self):
        def extract_price_info(text):
            price_per_night = nights = cleaning_fee = airbnb_service_fee = None
            if pd.notna(text):
                parts = text.split('****')
                for part in parts:
                    part = part.strip().lower()
                    if 'x' in part and 'nights' in part:
                        price_part, nights_part = part.split('x')
                        price_per_night = price_part.strip().replace('€', '').replace(',', '').strip()
                        nights = nights_part.split('nights')[0].strip().replace(',', '').strip()
                    if 'cleaning fee' in part and '€' in part:
                        cleaning_fee = part.split('€')[-1].strip().replace(',', '').strip()
                    if 'airbnb service fee' in part and '€' in part:
                        airbnb_service_fee = part.split('€')[-1].strip().replace(',', '').strip()
            return price_per_night, nights, cleaning_fee, airbnb_service_fee
    
        self.df[['Price_per_night', 'Nights', 'Cleaning_fee', 'Airbnb_service_fee']] = self.df['Additional_fees'].apply(
            lambda x: pd.Series(extract_price_info(x))
        ).fillna(0).astype(int)
        return self.df

    def cluster_and_rename_amenities(self):
        
        n_clusters=27
        split_amenities = self.df['Amenities'].str.split(r'\*\*\*\*').dropna().apply(lambda x: [item for item in x if not item.startswith('Unavailable:')])
    
        unique_amenities = list(set(itertools.chain.from_iterable(split_amenities)))

        model = SentenceTransformer('all-MiniLM-L6-v2')

        embeddings = model.encode(unique_amenities)

        clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = clustering_model.fit_predict(embeddings)

        clustered_amenities = {}
        for idx, cluster_id in enumerate(clusters):
            clustered_amenities.setdefault(cluster_id, []).append(unique_amenities[idx])

        amenity_to_cluster = {}
        for cluster_id, amenities in clustered_amenities.items():
            for amenity in amenities:
                amenity_to_cluster[amenity] = f"Cluster {cluster_id}"

        cluster_name_dict = {
            26: "Family Kitchen & Outdoor Dining",
            10: "Cooktops & Ovens",
            7: "Beachfront & Scenic Views",
            4: "Laundry Services & Equipment",
            12: "Fitness Facilities",
            15: "Parking Facilities",
            1: "Fridges & Cooling Devices",
            13: "Housekeeping Services",
            5: "Cooktops & Ovens",
            16: "Shampoo",
            23: "Outdoor Spaces",
            0: "Comfort Features & Bathing",
            8: "Pools",
            2: "Entertainment",
            22: "Baby Sleep Essentials",
            9: "BBQ Grills",
            18: "Soaps",
            25: "Conditioners",
            20: "Coffee Makers",
            3: "Smart TVs & Streaming",
            24: "TVs",
            14: "Cable TVs",
            11: "WiFi",
            6: "High Chairs and Child Seating",
            21: "Clothing Storage",
            19: "Sound Systems",
            17: "Children’s Books and Toys"
        }

        def rename_with_cluster_names(amenities_str, amenity_to_cluster, cluster_name_dict):
            if isinstance(amenities_str, str):
                segments = amenities_str.split('****')
                renamed_segments = []
                for segment in segments:
                    if segment in amenity_to_cluster:
                        cluster_id = int(amenity_to_cluster[segment].split()[-1])
                        renamed_segments.append(cluster_name_dict.get(cluster_id, segment))  
                    else:
                        renamed_segments.append(segment)  

                renamed_segments = list(set(renamed_segments))

                return renamed_segments
            else:
                return amenities_str

        self.df['renamed_amenities'] = self.df['Amenities'].apply(lambda x: rename_with_cluster_names(x, amenity_to_cluster, cluster_name_dict))
        self.df['renamed_amenities'] = self.df['renamed_amenities'].apply(
        lambda amenities: [item for item in amenities if not item.startswith("Unavailable:")] 
        if isinstance(amenities, list) else amenities
    )
    
        return self.df
        
    def parse_and_expand_comments(self):
        def parse_comments(comment_text, n_reviews):
            if not isinstance(comment_text, str):
                return [] 
                
            raw_comments = re.split(r'\*{2,}', comment_text)
            structured_comments = []

            for raw in raw_comments[-n_reviews:]:
                # Extract rating
                rating_match = re.search(r"Rating, (\d+) stars", raw)
                rating = int(rating_match.group(1)) if rating_match else None
                
                # Extract date
                date_match = re.search(r"(\w+ \d{4}|[\d]+ weeks ago|[\d]+ months ago|[\d]+ days ago)", raw)
                date = date_match.group(1).strip() if date_match else None
                
                # Extract duration
                duration_match = re.search(r"Stayed (.+)", raw)
                duration = duration_match.group(1).strip() if duration_match else None
                
                if rating or date or duration:
                    structured_comments.append({
                        'Rating': rating,
                        'Date': date,
                        'Duration': duration
                    })
            return structured_comments

        def convert_date(text_date):
            date = datetime(2024, 12, 9)
            try:
                if "weeks ago" in text_date:
                    weeks = int(re.search(r"(\d+) weeks ago", text_date).group(1))
                    return date - relativedelta(weeks=weeks)
                elif "months ago" in text_date:
                    months = int(re.search(r"(\d+) months ago", text_date).group(1))
                    return date - relativedelta(months=months)
                elif "days ago" in text_date:
                    days = int(re.search(r"(\d+) days ago", text_date).group(1))
                    return date - relativedelta(days=days)
                else:
                    return datetime.strptime(text_date, "%B %Y")
            except Exception:
                return None
    
        self.df['Comments_details'] = self.df['Comments_details'].fillna("").astype(str)
        
        self.df['Structured_Comments'] = self.df.apply(
        lambda row: parse_comments(row['Comments_details'], row.get('n_reviews', len(re.split(r'\*{2,}', row['Comments_details'])))),
        axis=1)
        
        expanded_comments = []
        for index, row in self.df.iterrows():
            for comment in row['Structured_Comments']:
                expanded_comments.append({
                    'ID': row['ID'],
                    **comment
                })

        comments_df = pd.DataFrame(expanded_comments)
        
        comments_df["Date"] = comments_df["Date"].apply(convert_date)
        comments_df["Date"] = comments_df["Date"].apply(
            lambda x: x.strftime("%B %Y") if pd.notna(x) else None
        )
        self.df = self.df.drop(columns='Structured_Comments')
        
        return comments_df


    def clean_airbnb_data(self):
        self.df = self.df.drop(columns='Unnamed: 0', errors='ignore')
        self.df = self.df.dropna(subset=['Title'])
        
        self.clean_reviews_rating()
        self.split_home_pieces()
        self.clean_price()
        self.cluster_and_rename_amenities()
        self.split_address()
        self.clean_country()
        self.clean_province()
        self.clean_city()
        self.clean_experience() 
        self.split_response_rate()
        self.split_additional_fees_column()
        comments_df = self.parse_and_expand_comments()
        self.df = self.df.drop(columns=['Address', 'Experience', 'Amenities', 'Home_pieces', 'Additional_fees', 'Comments_details', 'Response_rate', 'Rules', 'Country'])
        return self.df, comments_df

