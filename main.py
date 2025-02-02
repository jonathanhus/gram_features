import json
import sys
import os

from tap import Tap
from typing import Any, Dict, List, Literal, Optional, Union

import lingtypology.glottolog as glotto

from pathlib import Path
from tqdm import tqdm
from pathlib import Path
import sys

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


class Args(Tap):
    # language: Literal['ilo']
    language: str   # ISO Code for language
    use_rag: bool = False
    use_query_expansion: bool = False
    CHROMA_PATH: str = "./chroma_db"
    num_references: int = 2
    debug: bool = False
    model_version: str = "gpt-4o-2024-08-06"

args: Args = Args().parse_args()

# Function for querying the database
# def query_db(chain, search_string, use_rag):
#     if use_rag:
#         ref_data = chain.invoke(search_string)
#         return "To help answer the question, here is relevant data retrieved from a grammar book\n" + ref_data
#     else:
#         return ""

# Function for joining all the docs retrieved from vector store
def format_docs(docs):
    output = ""
    for doc in docs:
        output += str(doc.metadata) + "\n"
        output += doc.page_content + "\n"
    return output
    # for doc_set in doc_list:
    #     for doc in doc_set:
    #         output += str(doc.metadata) + "\n"
    #         output += doc.page_content + "\n"
    # return output

# Wrapper for glottocode function
# Modified to replace spaces with underscores
def get_by_iso(iso_code):
    name = glotto.get_by_iso(iso_code)
    return name.replace(" ", "_")


# Collect language identifiers
language = args.language
language_name = get_by_iso(language)
language_id = glotto.get_glot_id_by_iso(language)
print(f"Language: {language_name}")
print(language_id)

# Output
output_file = language.lower() + "_results"
if args.use_rag:
    output_file += "_" + str(args.num_references) + "_RAG"
if args.use_query_expansion:
    output_file += "_Query"
if args.debug:
    output_file += "_DEBUG"
output_file += ".json"

output_dir = Path('outputs') / language_name.lower()
output_path = output_dir / output_file

if output_path.is_file():
    print("Output already exists. Exiting now.")
    sys.exit()
else:
    print("Output file does not exist. Will create.")


# Load the features
feature_file = "resources/grambank/features/all_features.json"
with open(feature_file) as f:
    all_features = json.load(f)
list_of_features = list(all_features.keys())
if args.debug:
    list_of_features = ['GB020', 'GB051', 'GB522'] # smaller list for testing

# Define the model
model = ChatOpenAI(
    model=args.model_version,
    model_kwargs={ "response_format": {"type": "json_object"}},
)

system_template = (
    "You are an expert linguist with extensive knowledge about many languages."
    "Answer the following question about the language {language_name}. You are also provided "
    "with additional information about the question and you are given a procedure "
    "that indicates allowable answers for the question. You MUST provide an answer "
    "following the procedure. If you do not know the answer, answer 'IDK'."
    "Output the answer in JSON format with the following key-value pairs: "
    "'code': code, 'comment': other_data"
    "Please format the response as valid JSON that I can parse."
)

user_template = (
    "{question}\n"
    "Here is a summary about the question:\n"
    "{summary}\n"
    "{context}\n"  # Reference material retrieved from the vector database can go here
    "Here is the procedure to follow and the allowable responses:\n"
    "{procedure}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", user_template)]
)

# Define a model for paraphrasing
paraphrase_system_template = (
    "You are an expert at converting user questions into database queries."
    "Perform query expansion. If there are multiple common ways of phrasing "
    "a user question or common synonyms for key words in the question, make "
    "sure to return multiple versions of the query with the different "
    "phrasings. These are linguistic questions and additional information "
    "about the question is provided. Return 2 versions of the question and " 
    "summary combo in json format, where the "
    "keys are question_1 and question_2 and the value for each"
    "key is a json dict where the keys are question and summary"
)

paraphrase_user_template = (
    "Here is the question:\n"
    "{question}\n"
    "Here is the summary:\n"
    "{summary}"
)

paraphrase_prompt_template = ChatPromptTemplate.from_messages(
    [("system", paraphrase_system_template), ("user", paraphrase_user_template)]
)

expansion_chain = (
    paraphrase_prompt_template |
    model
)


# Test loading the database
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
embedding_db = Chroma(collection_name=language_name.lower(),
                      embedding_function=embeddings,
                      persist_directory=args.CHROMA_PATH
)

retriever = embedding_db.as_retriever(search_kwargs={'k': args.num_references})

# Define the rag_chain
rag_chain = retriever | format_docs

# Create the chain
chain = (
    prompt_template |
    model |
    SimpleJsonOutputParser()
)

# For Query Expansion, paraphrase the original question
def paraphrase_question(question, summary):
    questions = []
    paraphrase_model = (
        model |
        SimpleJsonOutputParser()
    )
    return questions


output_dict = dict()
print(f"{language_name} -- {language} -- {language_id}")
for feature in tqdm(list_of_features):
    paraphrase_list = []
    if args.use_query_expansion:
        query_data = {
            'question': all_features[feature]['feature'],
            'summary': all_features[feature]['Summary']
        }
        reworded_features = expansion_chain.invoke(query_data)
        reworded_features = json.loads(reworded_features.content)
        for reworded_feature in reworded_features:
            paraphrase_list.append(reworded_features[reworded_feature]['question'] + reworded_features[reworded_feature]['summary'])
    if args.use_rag or args.use_query_expansion:
        paraphrase_list.append(all_features[feature]['feature'] + all_features[feature]['Summary'])

        nested_documents_list = []
        for paraphrase in paraphrase_list:
            # nested_documents_list.append(retriever.invoke(all_features[feature]['feature'] + all_features[feature]['Summary']))
            nested_documents_list.append(retriever.invoke(all_features[feature]['feature']))
            # documents = retriever.invoke(all_features[feature]['feature'] + all_features[feature]['Summary'])
            # hus_doc = retriever.invoke(all_features[feature]['feature'] + all_features[feature]['Summary'])
        # flatten the list
        documents = [doc for sublist in nested_documents_list for doc in sublist]
        context_string = "To help answer the question, here is relevant data retrieved from a grammar book\n" + format_docs(documents)
    else:
        documents = ""
        context_string = ""

    example = {
        'language_name': language_name.capitalize(),
        'question': all_features[feature]['feature'],
        'summary': all_features[feature]['Summary'],
        'procedure': all_features[feature]['Procedure'],
        'context': context_string
        # 'context': query_db(rag_chain, all_features[feature]['feature'] + all_features[feature]['Summary'], args.use_rag)
    }
    value_id = f"{feature}-{language_id}"
    result = chain.invoke(example)
    # print(f"Result type: {type(result)}")
    # print(result)
    output_dict[value_id] = result
    output_dict[value_id]['source'] = [{'content': doc.page_content, 'metadata': doc.metadata} for doc in documents]
    output_dict[value_id]['prompt_info'] = example
    # output_dict[value_id]['source'] = example['context']


# Save results
output_path.parent.mkdir(exist_ok=True, parents=True)
with open(output_path, 'w') as f:
    json.dump(output_dict, f, indent=4, ensure_ascii=False)
print(f"Saved outputs to {output_path}")