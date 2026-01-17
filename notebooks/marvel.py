import pandas as pd
import os
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility


print("🔌 Подключение к Milvus...")
connections.connect("default", host="83.143.66.65", port="27370")


print("📥 Загрузка данных...")
df_faq = pd.read_excel("FAQ_for_RAG_Marvel.xlsx").fillna("") 

if "question" in df_faq.columns and "Question" not in df_faq.columns:
    df_faq["Question"] = df_faq["question"]
if "header" in df_faq.columns and "Header" not in df_faq.columns:
    df_faq["Header"] = df_faq["header"]


questions = df_faq["Question"].astype(str).tolist()
texts = df_faq["text"].astype(str).tolist()

document_names = df_faq["document_name"].astype(str).tolist() if "document_name" in df_faq.columns else [""] * len(questions)
headers = df_faq["Header"].astype(str).tolist() if "Header" in df_faq.columns else [""] * len(questions)


print("🧠 Загрузка модели intfloat/multilingual-e5-large...")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
model = SentenceTransformer("intfloat/multilingual-e5-large")
device = "cpu"
model = model.to(device)


print("🧪 Генерация эмбеддингов вопросов и текстов...")
question_embeddings = model.encode(
    questions,
    convert_to_tensor=True,
    show_progress_bar=True,
    device=device
).cpu().tolist()

text_embeddings = model.encode(
    texts,
    convert_to_tensor=True,
    show_progress_bar=True,
    device=device
).cpu().tolist()


collection_name = "faq_for_rag_e5_marvel"

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="document_name", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="header", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=16384),
    FieldSchema(name="qestion_e5", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="text_e5", dtype=DataType.FLOAT_VECTOR, dim=1024),
]
schema = CollectionSchema(fields, description="FAQ для RAG (Marvel, e5)")

print("\n🧾 Схема перед созданием коллекции:")
print(schema)

if utility.has_collection(collection_name):
    print(f"⚠️ Коллекция '{collection_name}' уже существует. Удаляем...")
    utility.drop_collection(collection_name)

collection = Collection(collection_name, schema=schema)


print("📤 Вставка документов, заголовков, вопросов и текстов в коллекцию...")
insert_result = collection.insert([
    document_names,
    headers,
    questions,
    texts,
    question_embeddings,
    text_embeddings
])
print(f"✅ Вставлено: {insert_result.insert_count} документов")

collection.flush()
print(f"\n📊 Фактически в коллекции: {collection.num_entities}")


print("⏳ Построение индексов...")
collection.create_index("qestion_e5", {"index_type": "AUTOINDEX", "metric_type": "COSINE"})
collection.create_index("text_e5", {"index_type": "AUTOINDEX", "metric_type": "COSINE"})

print("📦 Загружаем коллекцию в память...")
collection.load()

print(f"\n📊 Фактически в коллекции: {collection.num_entities}")