import streamlit as st
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.sql.functions import col, lower, regexp_replace, udf
from pyspark.sql.types import ArrayType, StringType
from nltk.stem import PorterStemmer
import nltk
import os

# Download nltk data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize Spark Session
if 'spark' not in st.session_state:
    st.session_state.spark = SparkSession.builder \
        .appName("HamOrSpamApp") \
        .master("local[*]") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()

spark = st.session_state.spark

# Load Model
@st.cache_resource
def load_model():
    # Path inside the container
    model_path = "/app/models/rf_model" 
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        return None
        
    try:
        model = RandomForestClassificationModel.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Preprocessing functions
def clean_text(df):
    df_normalized = df.withColumn(
        'text',
        lower(regexp_replace(regexp_replace("text", '[^a-zA-Z ]', ""), " +", " "))
    )
    return df_normalized

def tokenize_text(df):
    tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
    return tokenizer.transform(df)

def stem_text(df):
    stemmer = PorterStemmer()
    def stem_tokens(tokens_list):
        if tokens_list is None:
            return []
        return [stemmer.stem(w) for w in tokens_list]
    
    stem_tokens_udf = udf(stem_tokens, ArrayType(StringType()))
    return df.withColumn("stemmed_tokens", stem_tokens_udf("tokens"))

def vectorize_text(df):
    # HashingTF
    hashing_tf = HashingTF(
        inputCol="stemmed_tokens",
        outputCol="features",
        numFeatures=20000
    )
    df_tf = hashing_tf.transform(df)
    # idf = IDF(inputCol="raw_features", outputCol="features")
    # tf_idf_model = idf.fit(df_tf)
    # tf_idf = tf_idf_model.transform(df_tf) 
    
    st.warning("Using HashingTF only (IDF model missing). Predictions may be inaccurate.")
    # return df_tf.withColumnRenamed("raw_features", "features")
    return df_tf

st.title("Ham or Spam Classifier")

user_input = st.text_area("Enter email text:", height=200)

if st.button("Predict"):
    if user_input:
        # Create DataFrame
        data = [(user_input,)]
        df = spark.createDataFrame(data, ["text"])
        
        # Preprocess
        df_clean = clean_text(df)
        df_tokenized = tokenize_text(df_clean)
        df_stemmed = stem_text(df_tokenized)
        df_features = vectorize_text(df_stemmed)
        
        if model:
            try:
                predictions = model.transform(df_features)
                result = predictions.select("prediction").collect()[0]["prediction"]
                
                if result == 1.0:
                    st.error("SPAM Detected!")
                else:
                    st.success("This is HAM (Not Spam).")
            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        st.warning("Please enter some text.")
