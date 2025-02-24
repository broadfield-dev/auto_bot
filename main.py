from doitall import prompts
from doitall.addons import web_search as ws
from doitall.addons import file_handler as fh
from huggingface_hub import InferenceClient
import gradio as gr
import datetime
import requests
import random
import uuid
import json
import bs4
import os
import lxml
from pypdf import PdfReader

from langchain.embeddings import OpenAIEmbeddings 
from langchain.text_splitter import CharacterTextSplitter,RecursiveJsonSplitter
from langchain_huggingface import HuggingFaceEmbeddings

import chromadb
from chromadb.config import Settings
#from chromaviz import visualize_collection

from ollama import chat
from openai import OpenAI
from google import genai
from google.genai import types

from groq import Groq

import tiktoken
from dotenv import load_dotenv

load_dotenv()
hf_token=os.getenv('HF_KEY',None)
openai_key=os.getenv('OPENAI_API_KEY')
gemini_key=os.getenv('GEMINI_API_KEY')
groq_key = os.getenv('GROQ_API_KEY', None)

# Function to count tokens
def count_tokens(message):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Choose the appropriate model
    return len(encoding.encode(message))

class RagClient:
    def __init__(self, rag_dir="./RAG", col_rag='False'):
        try:
            self.clientDir = rag_dir
            print(self.clientDir)
            if not os.path.isdir(self.clientDir):os.mkdir(self.clientDir)
            #self.client = chromadb.CloudClient()
            if not col_rag:
                self.collection_name = 'memory'
            else:
                self.collection_name = col_rag
            self.model_list= ['sentence-transformers/all-mpnet-base-v2',
                              'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
                              'sentence-transformers/roberta-base-nli-mean-tokens',
                              'dunzhang/stella_en_400M_v5',
                              'shibing624/text2vec-base-chinese',
                              'sentence-transformers/stsb-roberta-base',
                              'dunzhang/stella_en_1.5B_v5',
                              'Alibaba-NLP/gte-multilingual-base',
                              'all-MiniLM-L6-v2',
                              ]
            self.model_name = self.model_list[0]
        except Exception as e:
            print(e)
          
    def read_pdf(pdf_path):
        if not os.path.isdir("./images"):
            os.mkdir("./images")
        text=[]
        images = ""
        reader = PdfReader(f'{pdf_path}')
        number_of_pages = len(reader.pages)
        file_name=str(pdf_path).split("\\")[-1]
        for i in range(number_of_pages):
            page = reader.pages[i]
            images=""
            if len(page.images) >0:
                for count, image_file_object in enumerate(page.images):
                    with open( "./images/" + str(count) + image_file_object.name, "wb") as fp:
                        fp.write(image_file_object.data)
                    #buffer = io.BytesIO(image_file_object.data)
                    #images.append({"name":file_name,"page":i,"cnt":count,"image":Image.open(buffer)})
                    #images.append(str(image_file_object.data))
                    fp.close()
                    images += "./images/" + str(count) + image_file_object.name + "\n"
            else:
                images=""
            text.append({"page":i,"text":page.extract_text(),"images":images})
        return text
    def save_memory(self,file_in,rag_col=""):
        print('save memory')
        self.collection_name=rag_col
        try:
            client = chromadb.PersistentClient(path=self.clientDir)
            if str(file_in).endswith(('.txt','.html','.json','.css','.js','.py','.svg')):
                with open(str(file_in), "r") as file:
                    document_text = file.read()
                file.close()
            elif str(file_in).endswith(('.pdf')):
                   pdf_json=self.read_pdf(str(file_in))
                   document_text='PDF_CONTENT: ' +json.dumps(pdf_json,indent=1)
            else: document_text = str(file_in);file_in="STRING"
            if type(document_text)==type({}):
                text_splitter = RecursiveJsonSplitter(chunk_size=1000, chunk_overlap=100)
                texts = text_splitter.split_json(document_text)
                print('recursive json split: ')
            else:    
                text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000, chunk_overlap=100)
                texts = text_splitter.split_text(document_text)
                print('character text split: ')

            emb_fn=HuggingFaceEmbeddings(model_name=self.model_name)
            collection = client.get_or_create_collection(
                name=self.collection_name, 
                embedding_function=emb_fn,
                metadata={
                    "hnsw:space": "cosine", #cosine, l2, ip
                    "hnsw:construction_ef": 100, #100 
                    "hnsw:search_ef": 100, #100
                    #hnsw:num_threads #default: multiprocessing.cpu_count()
                }  
            )
            doc_box=[]
            meta_d=[]
            ids=[]
            embeddings=emb_fn.embed_documents(texts=texts)

            for i,t in enumerate(texts):
                doc_box.append(t)
                meta_d.append({'timestamp':str(datetime.datetime.now()),'collection':self.collection_name,'filename':file_in,'chunk':i})
                ids.append(str(uuid.uuid4()))
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=meta_d,
                documents=doc_box,
            )
            print('memory saved to RAG')
            #print(self.collection)
        except Exception as e:
            print(e)

    def recall_memory(self, query_in, rag_col=""):
        self.collection_name=rag_col

        print('recall memory from RAG')
        print('QUERY IN: ', query_in)
        print('COLLECTION IN: ', rag_col)
        if not "COMPLETE" in str(query_in):
            emb_fn=HuggingFaceEmbeddings(model_name=self.model_name)
            embedded=emb_fn.embed_query(str(query_in))
            client = chromadb.PersistentClient(path=self.clientDir)
            
            collection = client.get_collection(
                name=self.collection_name, 
                embedding_function=emb_fn,
            )
            results = collection.query(
                query_embeddings=embedded,
                n_results=5,
            )
            print("Loaded")
            return results
        else: return ["No memories returned in response"]
    '''def view_collection(self,col_dir=""):
        emb_fn=HuggingFaceEmbeddings(model_name=self.model_name)
        client = chromadb.PersistentClient(path=self.clientDir)
        collection = client.get_collection(name=self.collection_name,embedding_function=emb_fn)
        visualize_collection(collection)'''
def isV(inp,is_=False,type=""):  # Verbose
    if is_==True:
        print(inp)
        is_=False
        

class Do_It_All:
    def __init__(self,clients,persist_dir="./db"):
        self.save_settings=[{}]
        self.merm_html="""**CONTENT**"""    
        self.html_html='''**CONTENT**"'''
        self.seed_val=1
        self.txt_clients = clients['txt']
        self.img_clients = clients['img']
        self.vis_clients = clients['vis']
        self.aud_clients = clients['aud']
        self.roles = []
        self.carry_hist = []
        self.collection_list=[]
        self.persist_dir=persist_dir

    def load_collections(self):
        client = chromadb.PersistentClient(path=self.persist_dir)
        choices = [list(x)[0][1] for x in client.list_collections()]
        return choices
    
    def view_collection(self,rag_col=""):
        rag=RagClient(rag_dir=self.persist_dir,col_rag=rag_col)
        rag.view_collection(col_dir=rag_col)
        
    def gen_im(self,prompt,seed, im_mod):
        isV('generating image', True)
        im_client=InferenceClient(self.img_clients[im_mod]['name'])
        image_out = im_client.text_to_image(prompt=prompt,height=256,width=256,num_inference_steps=10,seed=seed)
        output=f'{uuid.uuid4()}.png'
        image_out.save(output)
        isV(('Done: ', output), True)
        return [{'role':'assistant','content': {'path':output}}]

    def compress_data(self,c,purpose, history,mod,tok,seed,data):
        self.MAX_HISTORY=int(self.txt_clients[int(mod)]['max_tokens']) / 2

        isV(data)
        resp=[None,]
        #seed=random.randint(1,1000000000)
        isV (c)
        divr=int(c)/self.MAX_HISTORY
        divi=int(divr)+1 if divr != int(divr) else int(divr)
        chunk = int(int(c)/divr)
        isV(f'chunk:: {chunk}')
        isV(f'divr:: {divr}')
        isV (f'divi:: {divi}')
        task1="refine this data"
        out = []
        s=0
        e=chunk
        isV(f'e:: {e}')
        new_history=""
        data_l=["none",]
        task = f'Compile this data to fulfill the task: {task1}, and complete the purpose: {purpose}\n'
        for z in range(divi):
            data_l[0]=new_history
            isV(f's:e :: {s}:{e}')
            hist = history[s:e]
            resp = self.generate(
                prompt=purpose,
                history=hist,
                mod=int(mod),
                tok=int(tok),
                seed=int(seed),
                role='COMPRESS',
                data=data_l,
            )
            #resp_o = list(resp)[0]
            resp_o = resp
            new_history = resp_o
            isV (resp)
            out+=resp_o
            e=e+chunk
            s=s+chunk
        
        isV ("final" + resp_o)
        #history = [{'role':'system','content':'Compressed History: ' + str(resp_o)}]
        return str(resp_o)

    def find_all(self,prompt,history, url,mod,tok,seed,data):
        self.MAX_HISTORY=int(self.txt_clients[int(mod)]['max_tokens']) / 2
        return_list=[]
        isV (f"trying URL:: {url}")        
        try:
            if url != "" and url != None:    
                out = []
                headers = { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:82.0) Gecko/20100101 Firefox/82.0' } 
                source = requests.get(url, headers=headers)
                isV('status: ', source.status_code)
                if source.status_code ==200:
                    soup = bs4.BeautifulSoup(source.content,'lxml')
                    rawp=(f'RAW TEXT RETURNED: {soup.text}')
                    cnt=0
                    cnt+=len(rawp)
                    out.append(rawp)
                    out.append("HTML fragments: ")
                    q=("a","p","span","content","article")
                    for p in soup.find_all("a"):
                        out.append([{"LINK TITLE":p.get('title'),"URL":p.get('href'),"STRING":p.string}])
                    c=0
                    out2 = str(out)

                    if len(out2) > self.MAX_HISTORY:
                        isV("compressing...")
                        rawp = self.compress_data(len(out2),prompt,out2,mod,tok,seed, data)  
                    else:
                        isV(out)
                        rawp = out
                    return rawp
                else:
                    history.extend([{'role':'system','content':f"observation: That URL string returned an error: {source.status_code}, I should try a different URL string\n"}])
                    
                    return history
            else: 
                history.extend([{'role':'system','content':"observation: An Error occured\nI need to trigger a search using the following syntax:\naction: INTERNET_SEARCH action_input=URL\n"}])
                return history
        except Exception as e:
            isV (e)
            history.extend([{'role':'system','content':"observation: I need to trigger a search using the following syntax:\naction: INTERNET_SEARCH action_input=URL\n"}])
            return history

        
    def format_prompt(self,message, mod, system):
        eos=f"{self.txt_clients[int(mod)]['schema']['eos']}\n"
        bos=f"{self.txt_clients[int(mod)]['schema']['bos']}\n"
        prompt=""
        prompt+=bos
        prompt+=system
        prompt+=eos
        prompt+=bos
        prompt += message
        prompt+=eos
        prompt+=bos
        return prompt
    

    def format_ollama_prompt(self,message, mod, system):
        eos=f"{self.txt_clients[int(mod)]['schema']['eos']}\n"
        bos=f"{self.txt_clients[int(mod)]['schema']['bos']}\n"
        prompt=""
        prompt+=bos[0]
        prompt+=system
        prompt+=eos
        prompt+=bos[1]
        prompt += message
        prompt+=eos
        prompt+=bos[2]
        return prompt
    def llama_load():
        #llm = Llama.from_pretrained(
        #    repo_id="bullerwins/DeepSeek-V3-GGUF",
        #    filename="DeepSeek-V3-GGUF-bf16/DeepSeek-V3-Bf16-256x20B-BF16-00001-of-00035.gguf",
        # )
        pass
    def token_cost(self,input,in_tok,out_tok,ppt):
        if ppt == "None":
            input = "No 'Cost per Token'(tok_cost) vale found, cost set to zero"   
            price_per_token = 0         
        else:
            price_per_token = ppt
        total_tokens = in_tok + out_tok
        cost = total_tokens * price_per_token  # Replace price_per_token with actual price  
        tok_count= {
            'output': input,
            'tok_cost': price_per_token,
            'input_tokens': in_tok,
            'output_tokens': out_tok,
            'total_tokens': total_tokens,
            'cost': cost
        }
        return(json.dumps(tok_count,indent=4))

    def generate(self,prompt,history,mod=2,tok=4000,seed=1,role="RESPOND",data=None):
        print(seed)
        isV(role)
        hist_in=self.out_hist
        current_time=str(datetime.datetime.now())
        #timeline=str(data[4])
        self.roles=[{'name':'MANAGER','system_prompt':str(prompts.MANAGER.replace("**HISTORY**",str(hist_in)).replace("**ADVICE**",data[1].replace('<|im_start|>','').replace('<|im_end|>','')))},
                    {'name':'ADVISOR','system_prompt':str(prompts.ADVISOR.replace("**CURRENT_TIME**",current_time).replace("**TIMELINE**",str(data[4])).replace("**HISTORY**",str(history)))},
                    {'name':'PATHMAKER','system_prompt':str(prompts.PATH_MAKER.replace('**STEPS**',str(data[2])).replace("**CURRENT_OR_NONE**",str(data[4])).replace("**PROMPT**",json.dumps(data[0],indent=4)).replace("**HISTORY**",str(history)))},
                    {'name':'COMPRESS','system_prompt':str(prompts.COMPRESS.replace("**TASK**",str(prompt)).replace("**KNOWLEDGE**",str(data[0])).replace("**HISTORY**",str(history)))},
                    ]
        roles = self.roles
        g=True
        for roless in roles:
            if g==True:
                if roless['name'] == role:
                    system_prompt=roless['system_prompt']
                    isV(system_prompt)
                    g=False
                else: system_prompt = ""
                    
        
        if tok==None:isV('Error: tok value is None')
        isV("tok",tok)
        self.generate_kwargs = dict(
            temperature=0.99,
            max_new_tokens=tok, #total tokens - input tokens
            top_p=0.99,
            repetition_penalty=1.0,
            do_sample=True,
            seed=seed,
        )
        output = ""

        if self.txt_clients[int(mod)]['loc'] == 'hf':
            
            isV("Running ", self.txt_clients[int(mod)]['name'])
            if hf_token:
                self.client=InferenceClient(self.txt_clients[int(mod)]['name'],token=hf_token)
            else:
                return "You need an API key to do this"
            formatted_prompt = self.format_prompt(prompt, mod, system_prompt)
            stream = self.client.text_generation(formatted_prompt, **self.generate_kwargs, stream=True, details=True, return_full_text=True)
            if role in ['RESPOND','INTERNET_SEARCH']:
                for response in stream:
                    output += response.token.text
                yield output
            else:
                for response in stream:
                    output += response.token.text
                yield output
                
        elif self.txt_clients[int(mod)]['loc'] == 'ollama':
            isV("Running ", role)

            formatted_prompt = self.format_ollama_prompt(prompt, mod, system_prompt)
            stream = chat(
                model=self.txt_clients[int(mod)]['name'],
                messages=[{'role': 'system', 'content': system_prompt},{'role':'user','content':prompt}],
                stream=True,
            )

            for response in stream:
                output += response['message']['content']
                print(response['message']['content'], end='', flush=True)
            yield output    
            
        elif self.txt_clients[int(mod)]['loc'] == 'groq':
            #def stream_groq(model_name, messages):
            model=self.txt_clients[int(mod)]['name'],
            if not groq_key:
                yield hist_in+[{'role':'assistant','content':"Error: Groq API key not provided"}]
                return
            client = Groq(api_key=groq_key)
            stream = client.chat.completions.create(
                model=self.txt_clients[int(mod)]['name'],
                messages=[{'role': 'system', 'content': system_prompt},{'role':'user','content':prompt}],
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    output+=chunk.choices[0].delta.content
                    #yield hist_in+[{'role':'assistant','content':chunk.choices[0].delta.content.replace('<|im_start|>','').replace('<|im_end|>','')}]
            print(output)
            yield output
        
        elif self.txt_clients[int(mod)]['loc'] == 'openai':
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            stream = client.chat.completions.create(
            model=self.txt_clients[int(mod)]['name'],
            store=True,
            stream=True,
            messages=[{'role': 'system', 'content': system_prompt},{'role':'user','content':prompt}],

            )
            total_input_tokens = count_tokens(system_prompt) + count_tokens(prompt)  # Count tokens for input
            total_output_tokens = 0  # Initialize output token counter
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    output += chunk.choices[0].delta.content
            total_output_tokens += count_tokens(output)
            yield output
            tok_cnt = self.token_cost(output,total_input_tokens,total_output_tokens,self.txt_clients[int(mod)]['ppt'])


            self.carry_hist= hist_in+[{'role':'assistant','content':output + "\n\njson```\n" + tok_cnt + "\n```"}]
            #self.history += [{'role':'assistant','content':output}]


        elif self.txt_clients[int(mod)]['loc'] == 'google':
            isV("Running ", role)
            client = genai.Client(api_key=gemini_key)
            formatted_prompt = self.format_prompt(prompt, mod, system_prompt)

            response = client.models.generate_content_stream(
                model=self.txt_clients[int(mod)]['name'],
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature= 0.95,
                    max_output_tokens= tok,
                ),
            )
            total_input_tokens = count_tokens(system_prompt) + count_tokens(prompt)  # Count tokens for input
            total_output_tokens = 0  # Initialize output token counter
            for chunk in response:
                if type(chunk) != type(None):
                    output += chunk.text
            total_output_tokens += count_tokens(output)
            tok_cnt = self.token_cost(output,total_input_tokens,total_output_tokens,self.txt_clients[int(mod)]['ppt'])

            yield output
            yield history
            yield prompt

                   
    def generate_stream(self,prompt,history,mod=2,tok=4000,seed=1,role="RESPOND",data=None):
        print(seed)
        isV(role)
        hist_in=self.out_hist
        current_time=str(datetime.datetime.now())
        timeline=str(data[4])


        self.roles=[
                {'name':'INTERNET_SEARCH','system_prompt':str(prompts.INTERNET_SEARCH.replace("**TASK**",str(prompt)).replace("**KNOWLEDGE**",str(data[3])).replace("**HISTORY**",str(history)))},
                {'name':'RESPOND','system_prompt':str(prompts.RESPOND.replace("**CURRENT_TIME**",current_time).replace("**HISTORY**",str(history)).replace("**TIMELINE**",timeline))},
                ]
        roles = self.roles
        g=True
        for roless in roles:
            if g==True:
                if roless['name'] == role:
                    system_prompt=roless['system_prompt']
                    total_input_tokens = count_tokens(system_prompt) + count_tokens(prompt) # Count tokens for input
                    if tok-int(total_input_tokens * 1.5) < 0:
                        in_data=""
                        system_prompt = self.compress_data(len(str(system_prompt)),prompt, str(system_prompt),mod,6000,seed,in_data)
                    isV(system_prompt)
                    g=False
                else: system_prompt = ""
        total_input_tokens = (count_tokens(system_prompt) + count_tokens(prompt)) # Count tokens for input
        #if tok==None:isV('Error: tok value is None')
        tok=tok-int(total_input_tokens * 1.5)
        isV("tok",tok)
        self.generate_kwargs = dict(
            temperature=0.99,
            max_new_tokens=tok, #total tokens - input tokens
            top_p=0.99,
            repetition_penalty=1.0,
            do_sample=True,
            seed=seed,
        )
        output = ""
        total_output_tokens = 0  # Initialize output token counter
        if self.txt_clients[int(mod)]['loc'] == 'hf':
            
            isV("Running ", self.txt_clients[int(mod)]['name'])
            if hf_token:
                self.client=InferenceClient(self.txt_clients[int(mod)]['name'],token=hf_token)
            else:    
                return "You need an API key to do this"
            formatted_prompt = self.format_prompt(prompt, mod, system_prompt)
            stream = self.client.text_generation(formatted_prompt, **self.generate_kwargs, stream=True, details=True, return_full_text=True)
            if role in ['RESPOND','INTERNET_SEARCH']:
                for response in stream:
                    output += response.token.text
                    yield hist_in+[{'role':'assistant','content':output.replace('<|im_start|>','').replace('<|im_end|>','')}]
            else:
                for response in stream:
                    output += response.token.text
                    yield hist_in+[{'role':'assistant','content':output.replace('<|im_start|>','').replace('<|im_end|>','')}]
            self.carry_hist= hist_in+[{'role':'assistant','content':output.replace('<|im_start|>','').replace('<|im_end|>','')}]

        elif self.txt_clients[int(mod)]['loc'] == 'ollama':
            isV("Running ", role)

            formatted_prompt = self.format_ollama_prompt(prompt, mod, system_prompt)
            stream = chat(
                model=self.txt_clients[int(mod)]['name'],
                messages=[{'role': 'system', 'content': system_prompt},{'role':'user','content':prompt}],
                stream=True,
            )

            for response in stream:
                output += response['message']['content']
                #print(response['message']['content'], end='', flush=True)
                yield output

        elif self.txt_clients[int(mod)]['loc'] == 'groq':
            
            #def stream_groq(model_name, messages):
            model=self.txt_clients[int(mod)]['name'],
            if not groq_key:
                yield hist_in+[{'role':'assistant','content':"Error: Groq API key not provided"}]
                return
            client = Groq(api_key=groq_key)
            stream = client.chat.completions.create(
                model=self.txt_clients[int(mod)]['name'],
                messages=[{'role': 'system', 'content': system_prompt},{'role':'user','content':prompt}],
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    output+=chunk.choices[0].delta.content
                    yield hist_in+[{'role':'assistant','content':output.replace('<|im_start|>','').replace('<|im_end|>','')}]
            self.carry_hist= hist_in+[{'role':'assistant','content':output.replace('<|im_start|>','').replace('<|im_end|>','')}]

        
        elif self.txt_clients[int(mod)]['loc'] == 'openai':
            client = OpenAI(api_key=openai_key)
            stream = client.chat.completions.create(
            model=self.txt_clients[int(mod)]['name'],
            #store=True,
            stream=True,
            messages=[{'role': 'system', 'content': system_prompt},{'role':'user','content':prompt}],

            )
            
            total_input_tokens = count_tokens(system_prompt) + count_tokens(prompt)  # Count tokens for input
            total_output_tokens = 0  # Initialize output token counter
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    output += chunk.choices[0].delta.content
                    total_output_tokens += count_tokens(chunk.choices[0].delta.content)
                    yield hist_in+[{'role':'assistant','content':output}]
            
            
            total_output_tokens += count_tokens(output)
            tok_count = self.token_cost(output,total_input_tokens,total_output_tokens,self.txt_clients[int(mod)]['ppt'])

            yield hist_in+[{'role':'assistant','content':output}]+[{'role': 'assistant', 'content':f'```json\n{tok_count}\n```'}]
            self.carry_hist= hist_in+[{'role':'assistant','content':output}]
     
        elif self.txt_clients[int(mod)]['loc'] == 'google':
            output = ""
            total_output_tokens = 0

            client = genai.Client(api_key=gemini_key)
            #formatted_prompt = self.format_prompt(prompt, mod, system_prompt)

            response = client.models.generate_content_stream(
                model=self.txt_clients[int(mod)]['name'],
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature= 0.95,
                    max_output_tokens= tok,
                ),
            )

            total_input_tokens = count_tokens(system_prompt) + count_tokens(prompt)  # Count tokens for input
            total_output_tokens = 0  # Initialize output token counter

            for chunk in response:
                output += chunk.text
                total_output_tokens += count_tokens(chunk.text)
                yield hist_in + [{'role': 'assistant', 'content': output}]
 
            total_output_tokens += count_tokens(output)
            tok_count = self.token_cost(output,total_input_tokens,total_output_tokens,self.txt_clients[int(mod)]['ppt'])
            yield hist_in+[{'role':'assistant','content':output}]+[{'role': 'assistant', 'content':f'```json\n{tok_count}\n```'}]
            self.carry_hist= hist_in+[{'role':'assistant','content':output}]

    
    def multi_parse(self,inp):
        parse_boxes=[
            {'name':'json','cnt':7},
            {'name':'html','cnt':7},
            {'name':'css','cnt':6},
            {'name':'mermaid','cnt':10},
            ]
        isV("PARSE INPUT")
        isV(inp)
        if type(inp)==type(""):
            lines=""
            if "```" in inp:
                gg=True
                for ea in parse_boxes:
                    if gg==True:
                        if f"""```{ea['name']}""" in inp:
                            isV(f"Found {ea['name']} Code Block")
                            start = inp.find(f"```{ea['name']}") + int(ea['cnt'])  
                            end = inp.find("```", start) 
                            if start >= 0 and end >= 0:
                                inp= inp[start:end] 
                            else:
                                inp="NONE" 
                            isV("Extracted Lines")
                            isV(inp)
                            gg=False
                            return {'type':f"{ea['name']}",'string':str(inp)}
                        
            else:isV('ERROR: Code Block not detected')
        else:isV("ERROR: self.multi_parse requires a string input")

    def parse_from_str(self,inp):   
        rt=True
        out_parse={}
        for line in inp.split("\n"):
            if rt==True:
                if "```" in line:
                    out_parse=self.multi_parse(inp)
                    #rt=False
        if out_parse and out_parse['type']=='html':
            isV('HTML code: TRUE')
            html=self.html_html.replace('**CONTENT**',out_parse['string'].replace(","," ").replace("\n"," "))               
            #parse_url='https://'
        else:
            html=""
            isV('HTML code: TRUE')
        return html


    def parse_file_json(self,inp):
        isV("PARSE INPUT")
        isV(inp)
        if type(inp)==type(""):
            lines=""
            if "```json" in inp:
                start = inp.find("```json") + 7  
                end = inp.find("```", start) 
                if start >= 0 and end >= 0:
                    inp= inp[start:end] 
                else:
                    inp="NONE" 
                isV("Extracted Lines")
                isV(inp)
            try:
                out_json=eval(inp)
                out1=str(out_json['filename'])
                out2=str(out_json['filecontent'])
                return out1,out2
            except Exception as e:
                isV(e)
                return "None","None"
        if type(inp)==type({}):
            out1=str(inp['filename'])
            out2=str(inp['filecontent'])
            return out1,out2
    def read_pdf(self,pdf_path):
        if not os.path.isdir("./images"):
            os.mkdir("./images")
        text=[]
        images = ""
        reader = PdfReader(f'{pdf_path}')
        number_of_pages = len(reader.pages)
        file_name=str(pdf_path).split("\\")[-1]
        for i in range(number_of_pages):
            page = reader.pages[i]
            images=""
            if len(page.images) >0:
                for count, image_file_object in enumerate(page.images):
                    with open( "./images/" + str(count) + image_file_object.name, "wb") as fp:
                        fp.write(image_file_object.data)
                    #buffer = io.BytesIO(image_file_object.data)
                    #images.append({"name":file_name,"page":i,"cnt":count,"image":Image.open(buffer)})
                    #images.append(str(image_file_object.data))
                    images += "./images/" + str(count) + image_file_object.name + "\n"
                #text = f'{text}\n{page.extract_text()}'
            else:
                images=""
            text.append({"page":i,"text":page.extract_text(),"images":images})
        return text
    
    def save_file(self,input):
        if not os.path.isfile('./chat_dummp.txt'):
            out_chat=""
        else:
            with open('./chat_dump.txt','r') as f:
                out_chat=f.read()
            f.close()
        out_chat+=input[-1]['content'] + '/n'
        with open('./chat_dump.txt','w') as f:
            f.write(out_chat)   

    def agent(self,prompt_in,history,mod=2,im_mod=1,tok_in="",rand_seed=True,seed=1,max_thought=5,save_mem=False,recall_mem=False,rag_col=False):
        self.MAX_HISTORY=int(self.txt_clients[int(mod)]['max_tokens']) / 2
        self.out_hist=[]
        print(seed)
        isV(prompt_in,True)
        isV(('mod ',mod),True)
                
        merm="graph TD;A[Thought path...];"
        html=""
        in_data=["None","None","None","None","None","Mems","None"]
        prompt=prompt_in['text']
        fn=""
        com=""
        go=True
        save_to_file=False
        cnt=max_thought
        in_data[0]=prompt_in
        file_box=[]

        self.thought_hist=[{'role':'assistant','content':'Starting'}]
        yield self.out_hist + self.thought_hist

        if not history:history=[]
        history.extend([{'role':'user','content':prompt_in['text']}])
        self.out_hist=history.copy()
        self.thought_hist = [{'role':'assistant','content':'Starting RAG system...'}]
        ############################  Return Prompt #######################################
        yield self.out_hist

        ############################   Files In   #########################################
        for file_in in prompt_in['files']:
            self.thought_hist = [{'role':'assistant','content':'Loading user file...'}]
            yield self.out_hist + self.thought_hist
            if str(file_in).endswith(('.txt','.html','.json','.css','.js','.py','.svg')):
                if save_mem:
                    print("Saving file to RAG")
                    rag.save_memory(file_in=file_in,rag_col=rag_col)
                with open(str(file_in), "r") as file:
                    file_box.extend([{'doc_name':file_in,'content':file.read()}])
                    file.close()
            elif str(file_in).endswith(('.pdf')):
                pdf_json=self.read_pdf(str(file_in))
                file_box.extend([{'doc_name':file_in,'content':'PDF_CONTENT'+json.dumps(pdf_json,indent=1)}])    
                if save_mem:
                    rag.save_memory(file_in=json.dumps(file_box,indent=4),rag_col=rag_col)
        if len(str(file_box)) > self.MAX_HISTORY:
            file_out = self.compress_data(len(str(file_box)),prompt, str(file_box),mod,10000,seed,in_data)
        else: file_out=str(file_box)
        history.extend([{'role':'user','content':"Prompt: " +str(prompt) + "  File Content: " + str(file_out)}])
        self.thought_hist = [{'role':'assistant','content':'Starting main script...'}]
        yield self.out_hist + self.thought_hist     
        ############################   Starting Loop   #########################################
        while go == True:
            rag=RagClient(rag_dir=self.persist_dir,col_rag=rag_col)

            if recall_mem==True:
                self.thought_hist = self.out_hist + [{'role':'assistant','content':'Recalling memories...'}]
                yield self.thought_hist
                mems=rag.recall_memory(history, rag_col=rag_col)
            else:mems="No memories returned"


            in_data[5]=mems
            history.extend([{'role':'system','content':"RAG SYSTEM returned symantic search result : " + str(mems)}])
            max_tokens=int(self.txt_clients[int(mod)]['max_tokens'])

            #print(mems)
            if max_thought==0:
                in_data[2]="Unlimited"
            else:
                in_data[2]=cnt
            #isV(history)
            if rand_seed==True:
                seed = random.randint(1,99999999999999)
            else:
                seed = seed
            self.seed_val=seed
            c=0
            if len(str(history)) > self.MAX_HISTORY:
                #history = [{'role':'assistant','content':self.compress_data(len(str(history)),prompt,history,mod,2400,seed, in_data)  }]
                history = [{'role':'user','content':'USER prompt: ' + str(prompt_in)},{'role':'assistant','content':history[-2:] if len(str(history[-2:])) > self.MAX_HISTORY else history[-1:]}]
            isV('history',False)
            isV('calling PATHMAKER')
            role="PATHMAKER"
            #in_data[3]=file_list
            
            self.thought_hist = [{'role':'assistant','content':'Making Plan...'}]
            yield self.out_hist + self.thought_hist
            outph=self.generate(prompt,history,mod,2400,seed,role,data=in_data)
            path_out=list(outph)[0]
            out_parse={}
            rt=True
            for line in path_out.split("\n"):
                if rt==True:
                    if "```" in line:
                        out_parse=self.multi_parse(path_out)
                        #rt=False
                        if out_parse and out_parse['type']=='mermaid':
                            isV('Mermaid code: TRUE')
                            merm=self.merm_html.replace('**CONTENT**',out_parse['string'].replace(","," "))
                        elif out_parse and out_parse['type']=='html':
                            isV('HTML code: TRUE')
                            html=self.html_html.replace('**CONTENT**',out_parse['string'].replace(","," "))               
       
            self.thought_hist = [{'role':'assistant','content':'Choosing Path...'}]
            yield self.out_hist + self.thought_hist
            
            in_data[4]=str(merm)
            isV('calling ADVISOR')
            role="ADVISOR"
            advp=self.generate(prompt,history,mod,512,seed,role,in_data)
            advp_out=list(advp)[0]
            isV("HISTORY: ",history)
            isV('calling MANAGER')

            in_data[1]=advp_out
            role="MANAGER"
            outp=self.generate(prompt,history,mod,128,seed,role,in_data)
            
            outp0=list(outp)[0]
            isV(("Manager: ", outp0),True)
            #outp0 = re.sub('[^a-zA-Z0-9\s.,?!%()]', '', outpp)
            history.extend([{'role':'assistant','content':str(outp0)}])
            #yield history
            for line in outp0.split("\n"):
                if "action:" in line:
                    try:
                        com_line = line.split('action:')[1]
                        fn = com_line.split('action_input=')[0]
                        com = com_line.split('action_input=')[1].split('<|im_end|>')[0]
                        isV(com)
                        self.thought_hist = [{'role':'assistant','content':f'Calling command: {fn}'}]
                        self.thought_hist = [{'role':'assistant','content':f'Command input: {com}'}]
                   
                    except Exception as e:
                        if 'COMPLETE' in line:
                            isV('COMPLETE',True)
                            history.extend([{'role':'system','content':'Complete'}])
                            self.thought_hist = [{'role':'assistant','content':'Complete'}]
                            print("COMPLETE")
                            print(self.out_hist)
                            yield self.out_hist + self.thought_hist
                            go=False
                            break
                        else:fn="NONE"
                    
                    if 'RESPOND' in fn:
                        isV("RESPOND called")
                        self.thought_hist = [{'role':'assistant','content':'Formulating Response...'}]
                        yield self.out_hist + self.thought_hist

                        in_data[6]=self.out_hist
                        #print(self.out_hist)
                        
                        yield from self.generate_stream(prompt, history,mod,max_tokens,seed,role='RESPOND',data=in_data)

                       
                        history.extend(self.carry_hist[-1:])
                        history.extend([{'role':'assistant','content':'We just successfully completed the tool call RESPOND, now call: COMPLETE'}])
                        #print('self.out_hist, ,', self.out_hist)
                        self.out_hist=self.carry_hist
                        history+=self.carry_hist
                        if save_to_file:
                            self.save_file(self.out_hist)

                        if save_mem:
                            print("Saving RAG")
                            print(self.carry_hist)
                            rag.save_memory(file_in=self.carry_hist,rag_col=rag_col)
                        self.thought_hist = [{'role':'assistant','content':f'observation: We have used more than the Max Thought Limit, ending chat'}]
                        #yield self.out_hist
                        go=False
                        break
                    elif 'IMAGE' in fn:
                        self.thought_hist = [{'role':'assistant','content':'Generating Image...'}]
                        yield self.out_hist + self.thought_hist
                        isV('IMAGE called',True)
                        prompt_im=com
                        out_im=self.gen_im(prompt_im,seed, im_mod)
                        self.out_hist.extend(out_im)
                        #print(self.out_hist)
                        yield self.out_hist
                        history+=[{'role':'assistant','content':'Image Generation Completed Successfully'}]

                    elif 'INTERNET_SEARCH' in fn:
                        self.thought_hist = [{'role':'assistant','content':'Researching Topic...'}]
                        yield self.out_hist + self.thought_hist

                        isV('INTERNET_SEARCH called',True)
                        
                        if str(com).endswith(('.pdf')):
                            pdf_json = fh.download_pdf_to_bytesio(str(com))
                            #pdf_json=self.read_pdf(str(com))
                            ret = [{'doc_name':com,'PDF_CONTENT':json.dumps(pdf_json,indent=1)}]
                            print(ret)
                        else:
                            ret = ws.run_o(url=com)

                        in_data[3]=str(list(ret)[0])
                        self.thought_hist = [{'role':'assistant','content':'Compiling Report...'}]
                        yield self.out_hist + self.thought_hist
                        history.extend(self.carry_hist)
                        history.extend([{'role':'system','content':'thought: I have responded with a report of my recent internet search, this step is COMPLETE'}])
                        in_data[6]=self.out_hist
                        yield from self.generate_stream(prompt, history,mod,max_tokens,seed,role='INTERNET_SEARCH',data=in_data)
                        history.extend(self.carry_hist)
                        self.out_hist=self.carry_hist
                        if save_to_file:
                            self.save_file(self.out_hist)
                        if save_mem:
                            print("Saving RAG")
                            rag.save_memory(file_in=self.out_hist,rag_col=rag_col)
                    elif 'COMPLETE' in fn:
                        isV('COMPLETE',True)
                        history.extend([{'role':'system','content':'Complete'}])
                        self.thought_hist = [{'role':'assistant','content':'Complete'}]
                        print("COMPLETE")
                        print(self.out_hist)
                        yield self.out_hist + self.thought_hist

                        go=False
                        break

                    elif 'NONE' in fn:
                        isV('ERROR ACTION NOT FOUND',True)
                        history.extend([{'role':'assistant','content':f'observation:The last thing we attempted resulted in an error, check formatting on the tool call'}])
                   
                    else:
                        history.extend([{'role':'assistant','content':'observation: The last thing I tried resulted in an error, I should try selecting a different available tool using the format, action:TOOL_NAME action_input=required info'}])
                        pass;seed = random.randint(1,9999999999999)
                if save_mem==True:
                    self.thought_hist = self.out_hist + [{'role':'assistant','content':'Saving memories...'}]
                    yield self.thought_hist
                    rag.save_memory(file_in=history, rag_col=rag_col)
            if max_thought > 0:
                cnt-=1
                if cnt <= 0:
                    self.thought_hist = [{'role':'assistant','content':f'observation: We have used more than the Max Thought Limit, ending chat'}]
                    #yield self.out_hist
                    go=False
                    break
