"""
analyser.py — ResumeAI Core Engine v3
=======================================
All business logic. Zero UI code. Can be called from Streamlit,
FastAPI, CLI, or unit tests without modification.

What changed in v3:
  • ExperienceDomainAnalyser — extracts domain of experience
    and evaluates relevance to JD (not just years)
  • CertificationAnalyser    — rates cert relevance to JD,
    groups certs by domain, gives "Cert Verdict"
  • PatternDetector          — identifies overqualification,
    underqualification, domain mismatch, freshers
  • RecruiterVerdict         — explicit YES / MAYBE / NO with
    confidence level and reasoning chain
  • ATSScorer v3             — Projects weight raised to 15%,
    Education lowered to 5%, Experience stays 25%,
    Skills stays 40%, Keywords stays 15%
  • SimilarityEngine v3      — tri-gram TF-IDF + Jaccard +
    synonym-expanded overlap. Partial matches now also scored
    at the KEYWORD level (not just skill level).
"""

import re
import io
import warnings
from collections import Counter

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Optional

warnings.filterwarnings("ignore")

for _pkg in ["stopwords", "punkt", "wordnet", "punkt_tab"]:
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

_SW = set(stopwords.words("english"))
_LM = WordNetLemmatizer()

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════

SYNONYMS: dict[str, str] = {
    # AI / ML
    "ml": "machine learning", "ai": "artificial intelligence",
    "dl": "deep learning", "nlp": "natural language processing",
    "cv": "computer vision", "llm": "large language model",
    "llms": "large language model", "genai": "generative ai",
    "gen ai": "generative ai", "rl": "reinforcement learning",
    "xgb": "xgboost",
    # Cloud
    "gcp": "google cloud platform", "aws": "amazon web services",
    "k8s": "kubernetes", "tf": "terraform",
    "iac": "infrastructure as code",
    # Languages
    "js": "javascript", "ts": "typescript", "py": "python",
    "cplusplus": "c++", "c plus plus": "c++",
    "csharp": "c#", "dotnet": ".net", "golang": "go",
    # DB
    "pg": "postgresql", "postgres": "postgresql",
    "mssql": "sql server", "nosql": "nosql databases",
    # Web
    "nodejs": "node.js", "reactjs": "react",
    "vuejs": "vue", "nextjs": "next.js",
    # DevOps
    "cicd": "ci/cd", "ci cd": "ci/cd",
    # Data
    "bi": "business intelligence", "etl": "etl pipelines",
    "eda": "exploratory data analysis",
    "ab testing": "a/b testing", "abt": "a/b testing",
    # Engineering abbrevs
    "pm": "project management",
    "sdlc": "software development lifecycle",
    "oop": "object oriented programming",
    "oops": "object oriented programming",
    "restful": "rest api", "rest": "rest api",
    "graphql": "graphql api",
    # Roles
    "swe": "software engineer",
    "sde": "software development engineer",
    "mle": "machine learning engineer",
    "ds": "data scientist", "da": "data analyst",
    "ba": "business analyst", "qa": "quality assurance",
    "sre": "site reliability engineering",
    # Misc
    "rdbms": "relational database",
    "agile scrum": "agile",
    "ux": "user experience", "ui": "user interface",
    "api": "api development",
    "poc": "proof of concept", "mvp": "minimum viable product",
    "saas": "software as a service",
    "paas": "platform as a service",
    "faas": "function as a service",
    # Web / JS ecosystem
    "js": "javascript", "ts": "typescript", "rb": "ruby",
    "gql": "graphql api", "spa": "single page application",
    "ssr": "server side rendering", "cdn": "content delivery network",
    # Cloud specific services (map to parent)
    "eks": "kubernetes", "gke": "kubernetes", "aks": "kubernetes",
    "rds": "sql", "emr": "apache spark", "glue": "etl pipelines",
    # Monitoring / Observability
    "elk": "elasticsearch", "grafana": "monitoring", "prometheus": "monitoring",
    "datadog": "monitoring", "splunk": "siem", "pagerduty": "incident response",
    # Security
    "oauth2": "authentication", "oidc": "authentication",
    "saml": "authentication", "jwt": "authentication",
    "ssl": "cryptography", "tls": "cryptography",
    # IaC / Build
    "ansible": "infrastructure as code", "chef": "infrastructure as code",
    "puppet": "infrastructure as code", "make": "build tools",
    # Test automation
    "selenium": "test automation", "cypress": "test automation",
    "playwright": "test automation",
    # API / Integration
    "postman": "rest api", "swagger": "rest api", "openapi": "rest api",
    "websocket": "real time communication",
    # Data pipeline tools (alias to their parent)
    "luigi": "data pipeline", "prefect": "data pipeline",
    "flink": "apache spark", "presto": "sql", "trino": "sql",
    "clickhouse": "sql", "zookeeper": "distributed systems",
    "kafka": "event streaming", "rabbitmq": "message queue",
    "memcached": "nosql databases",
    # Infra
    "nginx": "web server", "haproxy": "load balancer",
    "traefik": "load balancer", "consul": "service mesh",
    "vault": "secret management", "envoy": "service mesh",
    "linkerd": "service mesh",
    # ML/AI model names → parent
    "gpt": "natural language processing",
    "gpt4": "natural language processing",
    "chatgpt": "natural language processing",
    "llama": "natural language processing",
    "mistral": "natural language processing",
    "gemini": "natural language processing",
    "stable diffusion": "computer vision",
    "whisper": "natural language processing",
    "chroma": "natural language processing",
    "pinecone": "natural language processing",
    "weaviate": "natural language processing",
    "qdrant": "natural language processing",
    "wandb": "mlops",
    # Viz tools
    "d3": "data visualization", "plotly": "data visualization",
    "bokeh": "data visualization", "looker": "data visualization",
    "metabase": "data visualization", "superset": "data visualization",
    # CRM / ERP / Finance
    "sfdc": "salesforce", "hubspot": "digital marketing",
    "marketo": "digital marketing", "mailchimp": "digital marketing",
    "ga4": "google analytics", "gtm": "google analytics",
    "quickbooks": "accounting", "tally": "accounting",
    "xero": "accounting", "netsuite": "accounting",
    "qbo": "accounting",
    # Misc
    "tdd": "test driven development", "bdd": "behavior driven development",
    "ddd": "domain driven design",
}

SKILL_HIERARCHY: dict[str, list[str]] = {
    "Python":           ["numpy","pandas","matplotlib","seaborn","scikit-learn",
                         "tensorflow","pytorch","keras","flask","django","fastapi",
                         "pytest","scipy","opencv","nltk","spacy","huggingface",
                         "streamlit","celery","sqlalchemy","beautifulsoup","scrapy",
                         "boto3","asyncio","pydantic","polars"],
    "Java":             ["spring","spring boot","hibernate","maven","gradle",
                         "junit","jackson","jpa","microservices","jdbc"],
    "JavaScript":       ["react","angular","vue","next.js","node.js","express",
                         "typescript","webpack","jest","jquery","redux",
                         "graphql api","tailwind","bootstrap","vite","nestjs"],
    "TypeScript":       ["react","angular","next.js","nestjs","deno","zod","prisma"],
    "C++":              ["stl","boost","qt","opencv","cmake","opengl","cuda",
                         "multithreading","memory management"],
    "C#":               [".net","asp.net","entity framework","wpf","unity",
                         "blazor","linq"],
    "Go":               ["goroutines","gin","grpc","protobuf","fiber"],
    "Rust":             ["cargo","actix","tokio","webassembly"],
    "Ruby":             ["rails","rspec","sinatra"],
    "PHP":              ["laravel","symfony","composer","wordpress"],
    "Swift":            ["swiftui","uikit","xcode","combine","coredata"],
    "Kotlin":           ["android","jetpack compose","coroutines","ktor"],
    "Scala":            ["akka","spark","sbt","cats"],
    "R":                ["ggplot2","dplyr","tidyr","shiny","caret","rmarkdown"],
    "Machine Learning": ["supervised learning","unsupervised learning",
                         "reinforcement learning","classification","regression",
                         "clustering","feature engineering","hyperparameter tuning",
                         "cross validation","xgboost","lightgbm","catboost",
                         "random forest","gradient boosting","model evaluation",
                         "scikit-learn","mlflow","model deployment"],
    "Deep Learning":    ["neural network","cnn","rnn","lstm","transformer",
                         "attention mechanism","backpropagation",
                         "batch normalization","dropout","transfer learning",
                         "object detection","image segmentation",
                         "diffusion models","vae"],
    "Natural Language Processing": [
                         "text classification","named entity recognition",
                         "sentiment analysis","word embedding","word2vec",
                         "bert","gpt","large language model","rag","fine tuning",
                         "prompt engineering","huggingface","spacy",
                         "langchain","llamaindex"],
    "Computer Vision":  ["image classification","object detection","yolo",
                         "opencv","image segmentation","face recognition",
                         "ocr","stable diffusion"],
    "Data Science":     ["exploratory data analysis","statistics",
                         "hypothesis testing","a/b testing",
                         "regression analysis","time series","forecasting",
                         "data cleaning","data wrangling","data visualization",
                         "pandas","numpy"],
    "MLOps":            ["mlflow","kubeflow","airflow","model serving",
                         "model monitoring","dvc","bentoml","seldon",
                         "feature store","data versioning"],
    "SQL":              ["mysql","postgresql","sqlite","oracle","sql server",
                         "stored procedures","indexing","query optimization",
                         "joins","transactions","normalization","window functions",
                         "cte"],
    "NoSQL Databases":  ["mongodb","cassandra","redis","dynamodb","couchdb",
                         "neo4j","firebase","elasticsearch","hbase"],
    "Data Warehouse":   ["snowflake","bigquery","redshift","databricks",
                         "synapse","hive","etl pipelines","elt","dbt",
                         "star schema","data modeling"],
    "Amazon Web Services": ["ec2","s3","lambda","rds","cloudformation",
                         "ecs","eks","sagemaker","glue","athena","cloudwatch",
                         "iam","vpc","api gateway","dynamodb","route53",
                         "cloudfront","step functions"],
    "Microsoft Azure":  ["azure vm","azure blob","azure functions",
                         "azure devops","cosmos db","azure kubernetes",
                         "azure ml","azure data factory","azure sql",
                         "azure active directory","power platform"],
    "Google Cloud Platform": ["compute engine","cloud storage","cloud functions",
                         "bigquery","vertex ai","cloud run","gke","pub/sub",
                         "dataflow","cloud spanner"],
    "DevOps":           ["ci/cd","jenkins","github actions","gitlab ci",
                         "argocd","infrastructure as code",
                         "site reliability engineering","monitoring","alerting",
                         "chaos engineering","observability"],
    "Docker":           ["dockerfile","docker compose","docker swarm",
                         "container registry","multi stage build","docker hub"],
    "Kubernetes":       ["pods","deployments","services","ingress","helm",
                         "kubectl","configmap","secrets","hpa","rbac",
                         "service mesh","istio","argo cd"],
    "Terraform":        ["hcl","modules","state management","providers",
                         "terragrunt"],
    "Linux":            ["bash","shell scripting","systemd","cron",
                         "networking","permissions","firewall","ssh","vim",
                         "process management","ubuntu","debian"],
    "Git":              ["github","gitlab","bitbucket","branching","merging",
                         "rebase","pull request","code review","git flow",
                         "monorepo"],
    "React":            ["hooks","redux","context api","react router","next.js",
                         "jsx","react native","testing library",
                         "styled components","zustand"],
    "Angular":          ["rxjs","ngrx","angular material","directives","pipes",
                         "signals"],
    "Node.js":          ["express","nestjs","fastify","socket.io","npm","yarn",
                         "pnpm","bun"],
    "Django":           ["django rest framework","orm","admin panel","celery",
                         "channels"],
    "Flask":            ["blueprints","sqlalchemy","jinja2","flask restful",
                         "marshmallow"],
    "REST API":         ["http methods","authentication","oauth","jwt",
                         "swagger","openapi","rate limiting",
                         "api versioning","api gateway"],
    "HTML/CSS":         ["semantic html","accessibility","responsive design",
                         "sass","scss","tailwind","bootstrap","css grid",
                         "flexbox","animations"],
    "Excel":            ["pivot table","vlookup","macros","vba","power query",
                         "formulas","conditional formatting"],
    "Tableau":          ["calculated fields","dashboard","data blending",
                         "lod","tableau server"],
    "Power BI":         ["dax","power query","report builder",
                         "power bi service","dataflows"],
    "Apache Spark":     ["pyspark","spark sql","spark streaming","rdd",
                         "spark mllib","delta lake"],
    "Apache Airflow":   ["dags","operators","sensors","hooks","scheduler",
                         "xcom"],
    "Financial Analysis": ["dcf","valuation","financial modeling",
                         "scenario analysis","lbo","comparable analysis"],
    "Accounting":       ["gaap","ifrs","balance sheet","income statement",
                         "cash flow","auditing","bookkeeping"],
    "SAP":              ["sap fi","sap co","sap mm","sap sd","sap hr",
                         "abap","sap hana","fiori"],
    "Risk Management":  ["credit risk","market risk","operational risk","var",
                         "stress testing","basel iii","regulatory compliance"],
    "Digital Marketing": ["seo","sem","ppc","google ads","facebook ads",
                          "email marketing","content marketing",
                          "social media marketing","google analytics",
                          "conversion optimization"],
    "UI/UX Design":     ["figma","sketch","adobe xd","prototyping",
                         "wireframing","usability testing","user research",
                         "design system","accessibility","interaction design"],
    "Agile":            ["scrum","kanban","sprint","backlog grooming",
                         "retrospective","story points","velocity",
                         "scrum master","product owner","jira"],
    "Project Management": ["jira","confluence","ms project","gantt",
                         "stakeholder management","pmp","prince2",
                         "risk management","resource planning","roadmap"],
    "Cybersecurity":    ["penetration testing","vulnerability assessment",
                         "owasp","network security","firewalls","siem","soc",
                         "incident response","cryptography","zero trust",
                         "threat modeling","security audit"],
    # ── New domains added in v3.1 ──
    "Test Automation":  ["selenium","cypress","playwright","jest","pytest",
                         "junit","mocha","jasmine","karma","testng","robot framework",
                         "appium","gatling","jmeter","locust","k6",
                         "unit testing","integration testing","e2e testing",
                         "tdd","bdd","test driven development","behavior driven development"],
    "Mobile Development": ["react native","flutter","swift","kotlin","android",
                           "ios","xcode","android studio","jetpack compose",
                           "swiftui","expo","ionic","cordova","xamarin","objective-c"],
    "Blockchain":       ["solidity","ethereum","web3","smart contracts","nft",
                         "defi","hardhat","truffle","metamask","chainlink",
                         "hyperledger","polygon","solana","rust"],
    "Data Analytics":   ["google analytics","mixpanel","amplitude","tableau",
                         "power bi","looker","metabase","superset","redash",
                         "excel","pivot table","sql","data studio",
                         "business intelligence","kpi","dashboard","reporting"],
    "Cloud Native":     ["kubernetes","docker","helm","istio","envoy","consul",
                         "vault","service mesh","api gateway","serverless",
                         "lambda","cloud run","cloud functions","azure functions",
                         "event driven","microservices","twelve factor"],
    "Event Streaming":  ["kafka","apache kafka","confluent","rabbitmq","activemq",
                         "redis streams","kinesis","pub/sub","event sourcing",
                         "cqrs","message queue","real time","stream processing"],
    "Embedded & IoT":   ["arduino","raspberry pi","rtos","microcontroller",
                         "firmware","fpga","vhdl","verilog","embedded c",
                         "iot","mqtt","zigbee","bluetooth","wifi","lora"],
    "Game Development": ["unity","unreal engine","godot","opengl","vulkan",
                         "directx","shader","glsl","hlsl","c++","c#","game design",
                         "physics engine","game mechanics","multiplayer"],
    "Bioinformatics":   ["biopython","bioinformatics","genomics","proteomics",
                         "r","next generation sequencing","blast","samtools",
                         "clinical trials","bioconductor","snp","gwas"],
    "Quantum Computing": ["qiskit","quantum circuit","quantum algorithm",
                          "quantum computing","qubit","entanglement",
                          "quantum machine learning","cirq","pennylane"],
    "AR/VR":            ["unity","unreal engine","arkit","arcore","vuforia",
                         "openxr","webxr","mixed reality","augmented reality",
                         "virtual reality","hololens","oculus","meta quest"],
    "Network Engineering": ["cisco","ccna","ccnp","bgp","ospf","mpls","vlan",
                             "tcp/ip","dns","dhcp","network security","firewall",
                             "load balancer","vpn","wan","lan","sd-wan",
                             "wireshark","packet analysis"],
    "Database Administration": ["mysql","postgresql","oracle","sql server","mongodb",
                                 "dba","database administration","backup recovery",
                                 "replication","partitioning","performance tuning",
                                 "query optimization","stored procedures","triggers"],
    "Content & Media":  ["adobe premiere","adobe after effects","final cut pro",
                         "davinci resolve","adobe photoshop","lightroom","illustrator",
                         "indesign","canva","figma","video editing","motion graphics",
                         "3d modeling","blender","maya","cinema 4d","zbrush"],
    "Supply Chain":     ["erp","sap","oracle scm","demand planning","inventory management",
                         "logistics","procurement","warehouse management",
                         "six sigma","lean manufacturing","kaizen","supplier management"],
    "Healthcare IT":    ["ehr","emr","epic","cerner","hl7","fhir","dicom",
                         "hipaa","healthcare analytics","clinical data","medical coding",
                         "icd-10","cpt","pharmacy systems","telehealth"],
    "Legal Tech":       ["contract management","ediscovery","legal research",
                         "compliance","gdpr","ccpa","intellectual property",
                         "case management","legal analytics","regulatory"],
    "Human Resources":  ["workday","bamboohr","successfactors","adp","payroll",
                         "recruitment","talent acquisition","onboarding",
                         "performance management","compensation","benefits",
                         "hris","learning management","employee engagement"],
    "Soft Skills":      ["leadership","communication","problem solving","teamwork",
                         "adaptability","critical thinking","creativity","time management",
                         "emotional intelligence","negotiation","presentation",
                         "conflict resolution","decision making","collaboration"],
}

ALL_SKILLS_FLAT: set[str] = set()
for _p, _ch in SKILL_HIERARCHY.items():
    ALL_SKILLS_FLAT.add(_p.lower())
    for _c in _ch:
        ALL_SKILLS_FLAT.add(_c.lower())

SECTION_KEYWORDS: dict[str, list[str]] = {
    "summary":        ["summary","objective","profile","about me","overview",
                       "professional summary","career objective"],
    "experience":     ["experience","work experience","employment history",
                       "career history","professional experience","work history",
                       "internship"],
    "education":      ["education","academic","qualification","degree",
                       "university","college","schooling"],
    "skills":         ["skills","technical skills","core competencies",
                       "expertise","key skills","proficiencies","technologies"],
    "projects":       ["projects","project work","personal projects",
                       "key projects","academic projects","open source",
                       "side projects"],
    "certifications": ["certification","certifications","licenses",
                       "credential","certified","accreditation"],
    "achievements":   ["achievement","accomplishment","award","honor",
                       "recognition","publications","research","patents",
                       "extra-curricular","activities"],
}

EDU_KEYWORDS: dict[str, int] = {
    "phd": 5, "ph.d": 5, "doctorate": 5,
    "master": 4, "mba": 4, "m.tech": 4, "msc": 4, "m.s": 4, "m.e": 4,
    "pg diploma": 3,
    "bachelor": 3, "b.tech": 3, "b.e": 3, "b.sc": 3, "bca": 2,
    "bba": 2, "b.com": 2,
    "diploma": 1, "associate": 1,
    "university": 1, "college": 1, "institute": 1,
}

EXPERIENCE_VERBS: list[str] = [
    "led","managed","built","developed","designed","implemented","deployed",
    "architected","created","launched","delivered","achieved","increased",
    "reduced","improved","optimised","automated","engineered","scaled",
    "refactored","migrated","integrated","coordinated","collaborated",
    "mentored","trained","supervised","directed","established","founded",
    "published","researched","analysed","evaluated","executed","drove",
    "spearheaded","oversaw","maintained","operated","negotiated",
    "presented","authored","streamlined","pioneered","transformed",
]

# Domain fingerprints — which skill clusters identify a domain
EXPERIENCE_DOMAINS: dict[str, list[str]] = {
    "Software Engineering":   ["python","java","javascript","typescript","c++","c#","go",
                                "rest api","microservices","git","docker","linux"],
    "Data Science / ML":      ["machine learning","deep learning","natural language processing",
                                "computer vision","data science","tensorflow","pytorch",
                                "pandas","numpy","scikit-learn","mlops"],
    "Data Engineering":       ["apache spark","apache airflow","etl pipelines","sql",
                                "data warehouse","snowflake","bigquery","databricks","dbt"],
    "Cloud / DevOps":         ["amazon web services","microsoft azure","google cloud platform",
                                "docker","kubernetes","terraform","devops","ci/cd",
                                "site reliability engineering"],
    "Frontend / UI":          ["react","angular","vue","html/css","javascript",
                                "typescript","ui/ux design","figma"],
    "Backend / Systems":      ["node.js","django","flask","rest api","sql",
                                "nosql databases","microservices","linux"],
    "Cybersecurity":          ["cybersecurity","penetration testing","owasp",
                                "network security","siem","incident response"],
    "Finance / Accounting":   ["financial analysis","accounting","sap","excel",
                                "risk management","sql"],
    "Product / Management":   ["agile","project management","jira","stakeholder management",
                                "product owner","scrum master"],
    "Marketing / Growth":     ["digital marketing","seo","google analytics",
                                "social media marketing","sem"],
    "Design":                 ["ui/ux design","figma","adobe xd","prototyping",
                                "user research","design system"],
}

# Cert domains and their relevance to job domains
CERT_DOMAINS: dict[str, dict] = {
    "Cloud & DevOps": {
        "keywords": ["aws certified","azure certified","google cloud certified",
                     "solutions architect","cloud practitioner","devops professional",
                     "kubernetes administrator","cka","ckad","terraform associate",
                     "aws saa","aws developer","azure administrator"],
        "relevant_to": ["Cloud / DevOps","Software Engineering","Backend / Systems",
                        "Data Engineering","Data Science / ML"],
    },
    "Data & AI": {
        "keywords": ["tensorflow developer","google data analytics","ibm data science",
                     "databricks certified","snowflake certified","deeplearning.ai",
                     "tableau desktop","power bi certified","microsoft data analyst",
                     "azure data scientist"],
        "relevant_to": ["Data Science / ML","Data Engineering"],
    },
    "Cybersecurity": {
        "keywords": ["cissp","ceh","comptia security","certified ethical hacker",
                     "oscp","security+","network+","cism","cisa","comptia cysa"],
        "relevant_to": ["Cybersecurity","Cloud / DevOps"],
    },
    "Project Management": {
        "keywords": ["pmp","prince2","agile certified","csm","safe","six sigma",
                     "lean six sigma","black belt","green belt","scrum master certified"],
        "relevant_to": ["Product / Management","Software Engineering",
                        "Finance / Accounting"],
    },
    "Finance": {
        "keywords": ["cfa","cpa","ca","acca","cma","frm","cia","chartered accountant",
                     "cfa level","certified public accountant"],
        "relevant_to": ["Finance / Accounting"],
    },
    "Microsoft / Office": {
        "keywords": ["microsoft certified","azure fundamentals",
                     "microsoft office specialist","mos","mcse","mcsa",
                     "power platform certified"],
        "relevant_to": ["Software Engineering","Finance / Accounting",
                        "Product / Management","Data Engineering"],
    },
    "Other Tech": {
        "keywords": ["cisco","ccna","ccnp","java certified","salesforce certified",
                     "hubspot certified","google analytics certified",
                     "google ads certified","meta certified"],
        "relevant_to": ["Software Engineering","Marketing / Growth",
                        "Backend / Systems"],
    },
}


# ═══════════════════════════════════════════════════════════════
# 1. TEXT CLEANER
# ═══════════════════════════════════════════════════════════════
class TextCleaner:
    """Normalise raw text for analysis. Tech tokens are preserved."""

    _KEEP = {"not","no","c","r","go","ml","ai","bi","qa","ui","ux","pm"}

    @classmethod
    def clean(cls, text: str, lemmatize: bool = True) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        text = re.sub(r"\S+@\S+", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[\+\(]?[0-9][\d\s\-\.]{7,}\d", " ", text)
        text = re.sub(r"c\+\+", "cplusplus", text, flags=re.IGNORECASE)
        text = re.sub(r"c#", "csharp", text, flags=re.IGNORECASE)
        text = re.sub(r"\.net\b", "dotnet", text, flags=re.IGNORECASE)
        text = re.sub(r"node\.js", "nodejs", text, flags=re.IGNORECASE)
        text = re.sub(r"next\.js", "nextjs", text, flags=re.IGNORECASE)
        text = re.sub(r"a/b\s*test", "ab testing", text, flags=re.IGNORECASE)
        text = re.sub(r"ci/cd", "cicd", text, flags=re.IGNORECASE)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        sw = _SW - cls._KEEP
        tokens = text.split()
        if lemmatize:
            tokens = [_LM.lemmatize(w) for w in tokens
                      if (w not in sw or w in cls._KEEP) and len(w) > 1]
        else:
            tokens = [w for w in tokens
                      if (w not in sw or w in cls._KEEP) and len(w) > 1]
        return " ".join(tokens)

    @classmethod
    def apply_synonyms(cls, text: str) -> str:
        t = " " + text.lower() + " "
        for abbr, canonical in SYNONYMS.items():
            t = re.sub(r"\b" + re.escape(abbr) + r"\b",
                       " " + canonical + " ", t)
        return re.sub(r"\s+", " ", t).strip()


# ═══════════════════════════════════════════════════════════════
# 2. SKILL EXTRACTOR
# ═══════════════════════════════════════════════════════════════
class SkillExtractor:
    """Fast substring skill extraction with synonym expansion."""

    @staticmethod
    def _prepare(text: str) -> str:
        t = TextCleaner.apply_synonyms(text.lower())
        t = " " + re.sub(r"[,;:()\[\]/\n\t\r\"']", " ", t) + " "
        return re.sub(r"\s+", " ", t)

    @classmethod
    def extract(cls, text: str) -> tuple[list[str], dict]:
        if not isinstance(text, str) or not text.strip():
            return [], {}
        t = cls._prepare(text)
        detected: set[str] = set()
        for skill in ALL_SKILLS_FLAT:
            if (" " + skill + " ") in t:
                detected.add(skill)
        hierarchy: dict = {}
        for parent, children in SKILL_HIERARCHY.items():
            pl = parent.lower()
            found_ch = [c for c in children if c in detected]
            if pl in detected or found_ch:
                hierarchy[parent] = {
                    "detected": pl in detected,
                    "children": found_ch,
                }
                detected.add(pl)
        return list(detected), hierarchy

    @classmethod
    def extract_jd_required(cls, jd_text: str) -> tuple[list[str], list[str]]:
        """Split JD skills into required vs preferred."""
        if not isinstance(jd_text, str):
            return [], []
        lines = jd_text.lower().split("\n")
        req_lines, pref_lines = [], []
        current = "required"
        pref_sigs = ["preferred","nice to have","good to have","plus","bonus",
                     "desirable","advantage","ideally","optional","preferred:"]
        req_sigs  = ["required","must have","mandatory","essential",
                     "minimum","need","required:","must:"]
        for line in lines:
            for s in pref_sigs:
                if s in line: current = "preferred"; break
            for s in req_sigs:
                if s in line: current = "required"; break
            (req_lines if current == "required" else pref_lines).append(line)
        req_sk,  _ = cls.extract(" ".join(req_lines))
        pref_sk, _ = cls.extract(" ".join(pref_lines))
        pref_only = [s for s in pref_sk if s not in set(req_sk)]
        return req_sk, pref_only


# ═══════════════════════════════════════════════════════════════
# 3. SIMILARITY ENGINE  (TF-IDF + Jaccard + keyword overlap)
# ═══════════════════════════════════════════════════════════════
class SimilarityEngine:
    """
    Hybrid semantic matching:
      35% TF-IDF cosine (bigrams + trigrams — captures "machine learning")
      40% Skill Jaccard with partial-match credit
      25% Synonym-expanded keyword overlap
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=20_000,
            sublinear_tf=True,
            min_df=1,
            stop_words="english",
        )

    def _fit_transform(self, texts: list[str]):
        cleaned = [TextCleaner.clean(TextCleaner.apply_synonyms(t)) for t in texts]
        try:
            return self._vectorizer.fit_transform(cleaned)
        except ValueError:
            return None

    def compute(self, resume_text: str, jd_text: str) -> dict:
        # Skill sets
        resume_skills, _ = SkillExtractor.extract(resume_text)
        jd_skills,     _ = SkillExtractor.extract(jd_text)
        res_set = set(resume_skills)
        jd_set  = set(jd_skills)
        matched = sorted(res_set & jd_set)
        missing = sorted(jd_set - res_set)

        # Partial matches — parent ↔ child upgrades
        partial: list[tuple[str, str]] = []
        still_missing: list[str] = []
        for jd_sk in missing:
            found = False
            for parent, children in SKILL_HIERARCHY.items():
                pl = parent.lower()

                # ensure it's always a list
                if isinstance(children, dict):
                    children = children.get("children", [])

                ch_lower = [c.lower() for c in children]
                if jd_sk in ch_lower and pl in res_set:
                    partial.append((pl, jd_sk))
                    found = True; break
                if jd_sk == pl:
                    match_ch = [c for c in ch_lower if c in res_set]
                    if match_ch:
                        partial.append((match_ch[0], jd_sk))
                        found = True; break
            if not found:
                still_missing.append(jd_sk)

        # Jaccard (partial = 0.5)
        jaccard_num = len(matched) + 0.5 * len(partial)
        skill_score = round(min(jaccard_num / max(len(jd_set), 1), 1.0) * 100, 1)

        # TF-IDF cosine
        tfidf_score = 0.0
        try:
            mx = self._fit_transform([resume_text, jd_text])
            if mx is not None:
                tfidf_score = round(float(cosine_similarity(mx[0], mx[1])[0][0]) * 100, 1)
        except Exception:
            pass

        # Keyword overlap (synonym-expanded)
        jd_exp  = set(TextCleaner.apply_synonyms(jd_text.lower()).split()) - _SW
        res_exp = set(TextCleaner.apply_synonyms(resume_text.lower()).split()) - _SW
        kw_ov   = len(jd_exp & res_exp) / max(len(jd_exp), 1)
        keyword_score = round(min(kw_ov, 1.0) * 100, 1)

        overall = round(
            0.35 * tfidf_score + 0.40 * skill_score + 0.25 * keyword_score, 1
        )
        overall = min(overall, 98.0)

        # Extra skills in resume not needed by JD
        extra = sorted(res_set - jd_set - set(s for _, s in partial))[:10]

        return {
            "overall":         overall,
            "tfidf_score":     tfidf_score,
            "skill_score":     skill_score,
            "keyword_score":   keyword_score,
            "matched_skills":  matched,
            "missing_skills":  still_missing,
            "partial_matches": partial,
            "extra_skills":    extra,
        }




# ═══════════════════════════════════════════════════════════════
# CONTACT VALIDATOR  (new in v3.1)
# ═══════════════════════════════════════════════════════════════
class ContactValidator:
    """
    Validates presence and format of contact information in a resume.
    Checks: email, phone, LinkedIn, GitHub, portfolio.
    Returns a structured dict with status and suggestions for each field.
    """

    _EMAIL_RE   = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
    _PHONE_RE   = re.compile(
        r"(?:\+?\d[\d\s\-\.\(\)]{7,15}\d"        # international / 10+ digit
        r"|\b[6-9]\d{9}\b"                               # Indian mobile
        r"|\b\d{3}[\-\.]\d{3}[\-\.]\d{4}\b)"     # US format
    )
    _LINKEDIN_RE = re.compile(
        r"(?:linkedin\.com/in/[\w\-]+|linkedin\.com/pub/[\w\-]+|linkedin:[\w\-]+)",
        re.IGNORECASE
    )
    _GITHUB_RE   = re.compile(
        r"(?:github\.com/[\w\-]+|github:[\w\-]+)", re.IGNORECASE
    )
    _PORTFOLIO_RE = re.compile(
        r"(?:behance\.net/|dribbble\.com/|kaggle\.com/|leetcode\.com/|"
        r"hackerrank\.com/|portfolio|my.*website|personal.*site)",
        re.IGNORECASE
    )

    @classmethod
    def validate(cls, text: str) -> dict:
        """
        Returns dict with keys: email, phone, linkedin, github, portfolio.
        Each value is: {found: bool, value: str|None, suggestion: str}
        Plus overall: {score: int/4, complete: bool}
        """
        if not isinstance(text, str):
            text = ""

        def _first(pattern):
            m = pattern.search(text)
            return m.group(0)[:60] if m else None

        email    = _first(cls._EMAIL_RE)
        phone    = _first(cls._PHONE_RE)
        linkedin = _first(cls._LINKEDIN_RE)
        github   = _first(cls._GITHUB_RE)
        portfolio= _first(cls._PORTFOLIO_RE)

        results = {
            "email": {
                "found": bool(email),
                "value": email,
                "label": "Email Address",
                "icon":  "📧",
                "suggestion": (
                    "" if email else
                    "Add a professional email address (e.g. firstname.lastname@gmail.com). "
                    "Avoid unprofessional usernames like 'coolboy123'."
                ),
            },
            "phone": {
                "found": bool(phone),
                "value": phone,
                "label": "Phone Number",
                "icon":  "📱",
                "suggestion": (
                    "" if phone else
                    "Add a phone number with country code (e.g. +91 9876543210 or +1 555-123-4567)."
                ),
            },
            "linkedin": {
                "found": bool(linkedin),
                "value": linkedin,
                "label": "LinkedIn Profile",
                "icon":  "💼",
                "suggestion": (
                    "" if linkedin else
                    "Add your LinkedIn URL (linkedin.com/in/yourname). "
                    "A complete LinkedIn profile increases recruiter response rate by 40%."
                ),
            },
            "github": {
                "found": bool(github),
                "value": github,
                "label": "GitHub Profile",
                "icon":  "🐙",
                "suggestion": (
                    "" if github else
                    "Add your GitHub URL (github.com/yourname). "
                    "Required for tech roles — recruiters check this before interviews."
                ),
            },
            "portfolio": {
                "found": bool(portfolio),
                "value": portfolio,
                "label": "Portfolio / Website",
                "icon":  "🌐",
                "suggestion": (
                    "" if portfolio else
                    "Add a portfolio link (Behance, Dribbble, personal site, Kaggle, LeetCode). "
                    "Optional but differentiates candidates."
                ),
            },
        }

        # Score: email + phone = required (2pts), linkedin + github = recommended (2pts)
        score = sum([
            1 if results["email"]["found"] else 0,
            1 if results["phone"]["found"] else 0,
            1 if results["linkedin"]["found"] else 0,
            1 if results["github"]["found"] else 0,
        ])
        results["score"] = score
        results["max_score"] = 4
        results["complete"] = score >= 3
        results["missing_critical"] = [
            k for k in ["email","phone"] if not results[k]["found"]
        ]

        return results


# ═══════════════════════════════════════════════════════════════
# CONTEXTUAL SKILL INFERENCER  (new in v3.1)
# ═══════════════════════════════════════════════════════════════
class ContextualSkillInferencer:
    """
    Infers skills that are implied by context but not explicitly stated.

    Examples:
      "Built a REST API deployed on AWS Lambda" implies: serverless, cloud, backend
      "Led a team of 8 engineers" implies: leadership, team management
      "Improved model accuracy from 78% to 94%" implies: machine learning, data science
      "Deployed microservices using Docker Compose" implies: docker, devops

    These are returned as "inferred_skills" with confidence levels (high/medium/low).
    They are NOT added to the main matched skills — only surfaced as suggestions
    that the candidate should explicitly state.
    """

    # Pattern → (implied_skills, confidence)
    _CONTEXT_PATTERNS: list[tuple] = [
        # Cloud / infra
        (r"deploy(?:ed|ing)\s+(?:to|on)\s+aws",            ["amazon web services","cloud"], "high"),
        (r"deploy(?:ed|ing)\s+(?:to|on)\s+(?:gcp|google cloud)", ["google cloud platform","cloud"], "high"),
        (r"deploy(?:ed|ing)\s+(?:to|on)\s+azure",          ["microsoft azure","cloud"], "high"),
        (r"serverless|lambda\s+function|cloud\s+function",  ["serverless","cloud"], "high"),
        (r"container(?:ised|ized)|docker(?:ise|ize)",         ["docker","devops"], "high"),
        (r"kubernetes|k8s\s+cluster|helm\s+chart",          ["kubernetes","devops"], "high"),
        (r"ci/cd|continuous\s+(?:integration|delivery|deployment)", ["ci/cd","devops"], "high"),
        (r"infra(?:structure)?\s+as\s+code|terraform|ansible", ["infrastructure as code","devops"], "high"),
        # ML / Data
        (r"train(?:ed|ing)\s+(?:a\s+)?model",               ["machine learning"], "high"),
        (r"model\s+accuracy|precision|recall|f1[\s\-]score", ["machine learning","data science"], "high"),
        (r"neural\s+net|deep\s+learning|backprop",           ["deep learning","machine learning"], "high"),
        (r"nlp\s+pipeline|text\s+classif|sentiment\s+anal", ["natural language processing"], "high"),
        (r"fine[\s\-]tun(?:ed|ing)\s+(?:llm|bert|gpt)",   ["natural language processing","deep learning"], "high"),
        (r"rag|retrieval\s+augmented",                        ["natural language processing"], "high"),
        (r"data\s+pipeline|etl\s+pipeline",                  ["etl pipelines","data engineering"], "high"),
        (r"feature\s+engineering|feature\s+extraction",     ["machine learning","data science"], "medium"),
        (r"a/b\s+test|hypothesis\s+test|statistical",        ["statistics","data science"], "medium"),
        (r"time\s+series|forecast(?:ed|ing)",                 ["time series","data science"], "medium"),
        # Backend / Systems
        (r"rest(?:ful)?\s+api|graphql\s+api|api\s+design", ["rest api"], "high"),
        (r"microservice|service[\s\-]oriented",              ["microservices"], "high"),
        (r"message\s+queue|event[\s\-]driven|pub[\s/\-]sub", ["event streaming"], "high"),
        (r"cache(?:d|ing)|redis|memcach",                      ["nosql databases","caching"], "medium"),
        (r"sql\s+query|database\s+optim|query\s+optim",    ["sql","query optimization"], "high"),
        (r"orm|object[\s\-]relational",                      ["sql"], "medium"),
        (r"authentication|authorisation|oauth|jwt\s+token",  ["authentication","rest api"], "medium"),
        # Frontend
        (r"react\s+component|hook|jsx|virtual\s+dom",        ["react"], "high"),
        (r"state\s+management|redux|zustand|mobx",            ["react","javascript"], "medium"),
        (r"responsive\s+design|mobile[\s\-]first|css\s+grid", ["html/css","responsive design"], "medium"),
        # Leadership / Process
        (r"led\s+(?:a\s+)?team|managing\s+(?:a\s+)?team|team\s+lead", ["leadership","team management"], "high"),
        (r"mentored|coaching\s+junior|code\s+review",        ["leadership","mentoring"], "medium"),
        (r"agile\s+sprint|scrum\s+ceremony|standup",         ["agile","scrum"], "high"),
        (r"roadmap|product\s+planning|stakeholder",           ["project management","product management"], "medium"),
        # Security
        (r"security\s+audit|penetration\s+test|pentest",    ["cybersecurity","penetration testing"], "high"),
        (r"owasp|xss|sql\s+injection|csrf",                   ["cybersecurity","owasp"], "high"),
        (r"encryption|hash(?:ing)?|ssl/tls",                   ["cryptography","cybersecurity"], "medium"),
        # Finance
        (r"financial\s+model|dcf|lbo\s+model",              ["financial analysis","financial modeling"], "high"),
        (r"p&l|profit\s+and\s+loss|income\s+statement",    ["accounting","financial analysis"], "high"),
        (r"variance\s+analysis|budget(?:ing)?|forecast",     ["financial analysis"], "medium"),
        # Design
        (r"figma|prototype|wireframe|user\s+flow",            ["ui/ux design","prototyping"], "high"),
        (r"usability\s+test|user\s+research|persona",        ["ui/ux design","user research"], "high"),
    ]

    @classmethod
    def infer(cls, resume_text: str, existing_skills: list[str]) -> list[dict]:
        """
        Scans resume for contextual signals and returns implied skills
        that the candidate has NOT explicitly stated.

        Returns list of dicts:
          {skill, confidence, evidence (the matching text), suggestion}
        """
        if not isinstance(resume_text, str):
            return []

        existing_lower = set(s.lower() for s in existing_skills)
        text_lower     = resume_text.lower()
        inferred: dict[str, dict] = {}

        for pattern, implied, confidence in cls._CONTEXT_PATTERNS:
            m = re.search(pattern, text_lower, re.IGNORECASE)
            if not m:
                continue
            # Get a snippet of the matching text for display
            start = max(0, m.start() - 20)
            end   = min(len(text_lower), m.end() + 40)
            snippet = resume_text[start:end].strip().replace("\n"," ")[:80]

            for skill in implied:
                if skill not in existing_lower and skill not in inferred:
                    inferred[skill] = {
                        "skill":      skill,
                        "confidence": confidence,
                        "evidence":   f"...{snippet}...",
                        "suggestion": (
                            f"Your resume mentions experience with '{skill.title()}' "
                            f"but doesn't list it explicitly. "
                            f"Add it to your Skills section to ensure ATS detection."
                        ),
                    }

        # Sort: high confidence first
        priority = {"high": 0, "medium": 1, "low": 2}
        return sorted(inferred.values(), key=lambda x: priority.get(x["confidence"],2))


# ═══════════════════════════════════════════════════════════════
# 4. SECTION PARSER
# ═══════════════════════════════════════════════════════════════
class SectionParser:
    """Structured extraction from resume text."""

    _DATE_RANGE = re.compile(
        r"(20\d\d|19\d\d)\s*[-–—to]+\s*(20\d\d|present|current|now)",
        re.IGNORECASE,
    )
    _EXP_EXPLICIT = [
        r"(\d+)\+?\s*years?\s*(?:of\s*)?(?:professional\s*)?experience",
        r"over\s*(\d+)\s*years?\s*(?:of\s*)?experience",
        r"(\d+)\s*years?\s*in\s+\w",
        r"(\d+)\s*yrs?\s*(?:of\s*)?(?:professional\s*)?experience",
    ]
    _COMPANY_WORDS = [
        "inc","ltd","pvt","llc","llp","corp","company","technologies",
        "solutions","systems","services","consulting","group","ventures",
        "labs","studio","foundation","bank","hospital","school","college",
        "university","institute","agency","firm","startup","tech",
    ]
    _ROLE_WORDS = [
        "engineer","developer","analyst","manager","lead","head","director",
        "architect","designer","consultant","officer","specialist","associate",
        "intern","scientist","researcher","coordinator","executive","programmer",
        "administrator","technician","devops","sre","mlops","data","software",
        "backend","frontend","fullstack","full stack","full-stack",
    ]

    @classmethod
    def detect_sections(cls, text: str) -> list[str]:
        t = text.lower()
        return [s for s, kws in SECTION_KEYWORDS.items() if any(k in t for k in kws)]

    @classmethod
    def extract_years(cls, text: str) -> int:
        if not isinstance(text, str): return 0
        nums: list[int] = []
        for pat in cls._EXP_EXPLICIT:
            for m in re.findall(pat, text.lower()):
                try: nums.append(int(m))
                except ValueError: pass
        for start, end in cls._DATE_RANGE.findall(text.lower()):
            try:
                s = int(start)
                e = 2025 if end.lower() in ["present","current","now"] else int(end)
                nums.append(max(0, e - s))
            except ValueError: pass
        return max(nums) if nums else 0

    @classmethod
    def education_score(cls, text: str) -> tuple[float, str]:
        if not isinstance(text, str): return 0.0, "Not detected"
        t = text.lower()
        best_s, best_d = 0, "Not detected"
        for kw, sc in EDU_KEYWORDS.items():
            if kw in t and sc > best_s:
                best_s = sc; best_d = kw.upper()
        return round(min(best_s / 5, 1.0), 3), best_d

    @classmethod
    def extract_experience_entries(cls, text: str) -> list[dict]:
        if not isinstance(text, str): return []
        entries: list[dict] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line or len(line) < 6: continue
            ll = line.lower()
            has_role    = any(rw in ll for rw in cls._ROLE_WORDS)
            has_company = any(cw in ll for cw in cls._COMPANY_WORDS)
            has_date    = bool(cls._DATE_RANGE.search(line) or
                               re.search(r"20\d\d|19\d\d", line))
            if not (has_role or has_company or has_date): continue
            entry: dict = {"raw": line[:120], "start": "", "end": "", "duration_yrs": 0}
            yr_m = cls._DATE_RANGE.search(line)
            if yr_m:
                entry["start"] = yr_m.group(1)
                end_s = yr_m.group(2)
                entry["end"] = "Present" if end_s.lower() in ["present","current","now"] else end_s
                try:
                    s = int(yr_m.group(1))
                    e = 2025 if end_s.lower() in ["present","current","now"] else int(end_s)
                    entry["duration_yrs"] = max(0, e - s)
                except ValueError: pass
            entries.append(entry)
            if len(entries) >= 8: break
        return entries

    @classmethod
    def has_quantified_achievements(cls, text: str) -> bool:
        if not isinstance(text, str): return False
        pats = [
            r"\d+\s*%", r"\$\s*\d+", r"\d+[kK]\+?",
            r"\d+\s*(?:million|billion|m|b)\b",
            r"\d+\s*(?:users?|customers?|clients?|team|people|employees?)",
            r"reduced\s.*?\d+", r"increased\s.*?\d+",
            r"improved\s.*?\d+", r"delivered\s.*?\d+",
            r"saved\s.*?\d+", r"generated\s.*?\d+",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in pats)

    @classmethod
    def count_action_verbs(cls, text: str) -> int:
        if not isinstance(text, str): return 0
        t = text.lower()
        return sum(1 for v in EXPERIENCE_VERBS
                   if re.search(r"\b" + v + r"\b", t))

    @classmethod
    def detect_projects(cls, text: str) -> tuple[bool, int]:
        """Returns (has_projects, project_count_estimate)."""
        if not isinstance(text, str): return False, 0
        t = text.lower()
        has = ("project" in t or "built" in t or "developed" in t or
               "github" in t or "deployed" in t or "open source" in t)
        # Rough count: lines that look like project titles
        count = len(re.findall(
            r"(?:project|application|system|tool|platform|bot|api|dashboard|app)[:\-–\s]+[a-z]",
            t))
        return has, min(count, 10)

    @classmethod
    def has_internship(cls, text: str) -> bool:
        if not isinstance(text, str): return False
        t = text.lower()
        return any(w in t for w in ["intern","internship","trainee",
                                     "apprentice","placement",
                                     "industrial training"])

    @classmethod
    def extract_online_presence(cls, text: str) -> dict:
        if not isinstance(text, str):
            return {"github": False, "linkedin": False, "portfolio": False}
        t = text.lower()
        return {
            "github":    "github" in t,
            "linkedin":  "linkedin" in t,
            "portfolio": any(w in t for w in ["portfolio","behance","dribbble",
                                               "kaggle","leetcode","hackerrank"]),
        }


# ═══════════════════════════════════════════════════════════════
# 5. EXPERIENCE DOMAIN ANALYSER  (NEW in v3)
# ═══════════════════════════════════════════════════════════════
class ExperienceDomainAnalyser:
    """
    Determines WHAT DOMAIN the candidate's experience is in,
    then evaluates whether that domain is relevant to the JD.

    Example output:
      candidate_domains: ["Data Science / ML", "Software Engineering"]
      jd_domains:        ["Data Science / ML", "Cloud / DevOps"]
      domain_match_pct:  67.0
      relevance_label:   "Partially Relevant"
      insight:           "2 of 3 JD domains match your experience. ..."
    """

    @staticmethod
    def _score_domain(text: str, skills: list[str]) -> dict[str, float]:
        """Return domain → strength score for a piece of text."""
        skill_set = set(s.lower() for s in skills)
        scores: dict[str, float] = {}
        for domain, fingerprints in EXPERIENCE_DOMAINS.items():
            hits = sum(1 for fp in fingerprints if fp in skill_set)
            if hits > 0:
                scores[domain] = round(hits / len(fingerprints), 3)
        return scores

    @classmethod
    def analyse(cls, resume_text: str, jd_text: str,
                resume_skills: list[str],
                jd_skills: list[str]) -> dict:
        res_scores = cls._score_domain(resume_text, resume_skills)
        jd_scores  = cls._score_domain(jd_text, jd_skills)

        # Top domains for each
        cand_domains = sorted(res_scores, key=res_scores.get, reverse=True)[:3]
        jd_domains   = sorted(jd_scores,  key=jd_scores.get,  reverse=True)[:3]

        # Domain overlap
        cand_set = set(cand_domains)
        jd_set   = set(jd_domains)
        overlap  = cand_set & jd_set
        domain_match_pct = round(
            len(overlap) / max(len(jd_set), 1) * 100, 1
        ) if jd_set else 0.0

        # Relevance label
        if domain_match_pct >= 66:
            rel_label = "Highly Relevant"
            rel_color = "#22c55e"
        elif domain_match_pct >= 33:
            rel_label = "Partially Relevant"
            rel_color = "#eab308"
        elif cand_domains:
            rel_label = "Domain Mismatch"
            rel_color = "#ef4444"
        else:
            rel_label = "Unclear Domain"
            rel_color = "#94a3b8"

        # Human insight
        if not jd_domains:
            insight = "Could not determine JD domain from the provided text."
        elif not cand_domains:
            insight = ("No clear domain detected in resume. "
                       "The resume may be too generic or lacks domain-specific keywords.")
        elif domain_match_pct >= 66:
            insight = (f"Your experience in {', '.join(cand_domains[:2])} strongly aligns "
                       f"with the JD's requirement for {', '.join(jd_domains[:2])}.")
        elif overlap:
            matched_str   = ", ".join(overlap)
            unmatched_str = ", ".join(jd_set - cand_set)
            insight = (f"Experience partially matches. Aligned domains: {matched_str}. "
                       f"JD also needs: {unmatched_str}. "
                       f"Highlight any relevant work in {'those areas' if unmatched_str else 'those domains'}.")
        else:
            insight = (f"Domain mismatch detected. Your experience is in "
                       f"{', '.join(cand_domains[:2])}, but the JD requires "
                       f"{', '.join(jd_domains[:2])}. "
                       "Consider reframing your experience to highlight transferable skills.")

        return {
            "candidate_domains":  cand_domains,
            "jd_domains":         jd_domains,
            "overlapping_domains":sorted(overlap),
            "domain_match_pct":   domain_match_pct,
            "relevance_label":    rel_label,
            "relevance_color":    rel_color,
            "insight":            insight,
            "res_scores":         res_scores,
            "jd_scores":          jd_scores,
        }


# ═══════════════════════════════════════════════════════════════
# 6. CERTIFICATION ANALYSER  (NEW in v3)
# ═══════════════════════════════════════════════════════════════
class CertificationAnalyser:
    """
    Detects certifications, groups them by domain, counts them,
    evaluates relevance to the JD, and gives a cert verdict.

    Verdict examples:
      "3 certs in Cloud & DevOps — Strongly supports this role"
      "2 certs in Finance — Not directly relevant to this ML role"
      "No certifications — Competitive candidates typically have 1-2"
    """

    @classmethod
    def extract(cls, text: str) -> tuple[dict, int]:
        if not isinstance(text, str): return {}, 0
        t = text.lower()
        found: dict = {}
        total = 0
        for domain, data in CERT_DOMAINS.items():
            hits: list[str] = []
            for kw in data["keywords"]:
                if re.search(r"\b" + re.escape(kw) + r"\b", t):
                    idx = t.find(kw)
                    snippet = re.sub(r"\s+", " ",
                                     text[max(0,idx-5):min(len(t),idx+70)].strip())
                    hits.append(snippet[:70])
            if hits:
                found[domain] = {"snippets": hits, "count": len(hits)}
                total += len(hits)
        return found, total

    @classmethod
    def analyse_relevance(cls, certs: dict, cert_count: int,
                           domain_result: dict) -> dict:
        """
        Cross-reference cert domains with JD domains to produce
        a relevance verdict and explanation.
        """
        jd_domains = set(domain_result.get("jd_domains", []))
        if cert_count == 0:
            return {
                "verdict":       "No Certifications",
                "verdict_color": "#94a3b8",
                "explanation":   ("No certifications detected. Certifications significantly "
                                  "differentiate candidates — consider adding 1-2 relevant ones."),
                "relevance_pct": 0,
                "relevant_domains": [],
                "irrelevant_domains": [],
            }

        relevant: list[str]   = []
        irrelevant: list[str] = []
        for cert_domain in certs:
            cd_data = CERT_DOMAINS.get(cert_domain, {})
            relevant_to = cd_data.get("relevant_to", [])
            if any(jd in relevant_to for jd in jd_domains) or not jd_domains:
                relevant.append(cert_domain)
            else:
                irrelevant.append(cert_domain)

        rel_pct = round(len(relevant) / max(cert_count, 1) * 100, 1)
        total_rel_certs = sum(
            certs[d]["count"] for d in relevant if d in certs
        )

        if total_rel_certs >= 2 and rel_pct >= 50:
            verdict = f"{total_rel_certs} Relevant Cert{'s' if total_rel_certs>1 else ''}"
            v_color = "#22c55e"
            expl = (f"{total_rel_certs} certification(s) in {', '.join(relevant[:2])} "
                    f"directly support this role. Strong credential signal.")
        elif total_rel_certs >= 1:
            verdict = "1 Relevant Cert"
            v_color = "#eab308"
            expl = (f"1 relevant certification found ({', '.join(relevant[:1])}). "
                    f"Adding 1-2 more targeted certs would significantly strengthen the profile.")
        elif cert_count > 0:
            verdict = "Certs Not Role-Relevant"
            v_color = "#f97316"
            expl = (f"{cert_count} certification(s) found but in {', '.join(irrelevant[:2])}, "
                    f"which are not directly relevant to this role. "
                    f"Consider adding certifications in {', '.join(list(jd_domains)[:2]) or 'the JD domain'}.")
        else:
            verdict = "No Certifications"
            v_color = "#94a3b8"
            expl = "No certifications detected."

        return {
            "verdict":            verdict,
            "verdict_color":      v_color,
            "explanation":        expl,
            "relevance_pct":      rel_pct,
            "relevant_domains":   relevant,
            "irrelevant_domains": irrelevant,
        }


# ═══════════════════════════════════════════════════════════════
# 7. PATTERN DETECTOR  (NEW in v3)
# ═══════════════════════════════════════════════════════════════
class PatternDetector:
    """
    Identifies high-level patterns that a recruiter would notice:
      - Overqualification (candidate seems senior for a junior role)
      - Underqualification (candidate seems junior for a senior role)
      - Domain mismatch (experience doesn't match JD domain)
      - Career change / pivot
      - Fresher / student profile
      - Strong but specialist (deep in one area)
      - Well-rounded generalist
    Each pattern has: label, severity (info/warn/ok), description.
    """

    @staticmethod
    def detect(
        resume_text: str,
        jd_text: str,
        yrs: int,
        skill_count: int,
        domain_result: dict,
        sections: list[str],
        cert_count: int,
        has_quant: bool,
        has_projects: bool,
        has_internship: bool,
    ) -> list[dict]:
        patterns: list[dict] = []
        jd_t = jd_text.lower()

        # ── Experience level vs JD expectation ──
        jd_wants_senior = any(w in jd_t for w in
            ["senior","sr.","lead","principal","staff","architect",
             "5+ years","5 years","7+ years","8+ years"])
        jd_wants_junior = any(w in jd_t for w in
            ["junior","jr.","entry","fresher","0-2 years","1-2 years",
             "graduate","intern","trainee","fresh","beginner"])

        if jd_wants_junior and yrs >= 5:
            patterns.append({
                "label": "Potential Overqualification",
                "severity": "warn",
                "description": (
                    f"Candidate has {yrs}+ years of experience, but the JD targets entry/junior level. "
                    "Recruiter may question retention and salary expectations."
                ),
                "icon": "⬆️",
            })
        elif jd_wants_senior and yrs <= 2:
            patterns.append({
                "label": "Potential Underqualification",
                "severity": "warn",
                "description": (
                    f"JD expects senior/lead level but candidate has only {yrs or '<1'} year(s). "
                    "Strong projects and certifications can partially compensate."
                ),
                "icon": "⬇️",
            })

        # ── Domain mismatch ──
        if domain_result.get("relevance_label") == "Domain Mismatch":
            cand_d = domain_result.get("candidate_domains", [])
            jd_d   = domain_result.get("jd_domains", [])
            patterns.append({
                "label": "Domain Mismatch",
                "severity": "warn",
                "description": (
                    f"Candidate's experience ({', '.join(cand_d[:2]) or 'unclear'}) "
                    f"differs from what the JD requires "
                    f"({', '.join(jd_d[:2]) or 'unclear'}). "
                    "Resume should be reframed to highlight transferable skills."
                ),
                "icon": "🔀",
            })

        # ── Fresher profile ──
        if yrs == 0 and not has_internship:
            patterns.append({
                "label": "Fresher / Student Profile",
                "severity": "info",
                "description": (
                    "No work experience or internship detected. "
                    "Candidate is likely a fresher. "
                    "Evaluate on projects, certifications, and academic background."
                ),
                "icon": "🎓",
            })
        elif yrs == 0 and has_internship:
            patterns.append({
                "label": "Internship Experience Only",
                "severity": "info",
                "description": (
                    "Candidate has internship experience but no full-time work history. "
                    "Good signal for entry-level roles."
                ),
                "icon": "🔰",
            })

        # ── Specialist vs generalist ──
        if skill_count >= 20:
            domain_scores = domain_result.get("res_scores", {})
            top_scores = sorted(domain_scores.values(), reverse=True)
            if top_scores and top_scores[0] >= 0.4 and len(top_scores) == 1:
                patterns.append({
                    "label": "Deep Specialist",
                    "severity": "ok",
                    "description": (
                        f"Candidate has {skill_count} skills concentrated in "
                        f"{domain_result['candidate_domains'][0]}. "
                        "Strong fit for specialist roles, may lack breadth for generalist ones."
                    ),
                    "icon": "🎯",
                })
            elif skill_count >= 25 and len(domain_scores) >= 4:
                patterns.append({
                    "label": "Well-Rounded Generalist",
                    "severity": "ok",
                    "description": (
                        f"Candidate spans {len(domain_scores)} domains with {skill_count} skills. "
                        "Versatile profile — good for roles requiring cross-functional collaboration."
                    ),
                    "icon": "🌐",
                })

        # ── No quantified achievements ──
        if not has_quant and yrs >= 2:
            patterns.append({
                "label": "Lacks Quantified Results",
                "severity": "warn",
                "description": (
                    "No metrics, percentages, or numbers detected in experience bullets. "
                    "Experienced candidates are expected to quantify their impact."
                ),
                "icon": "📊",
            })

        # ── Strong profile ──
        if yrs >= 3 and has_quant and cert_count >= 1 and skill_count >= 15:
            patterns.append({
                "label": "Strong Candidate Profile",
                "severity": "ok",
                "description": (
                    f"{yrs}+ years experience, {cert_count} certification(s), "
                    f"{skill_count} skills, and quantified achievements. "
                    "Well-rounded professional profile."
                ),
                "icon": "⭐",
            })

        return patterns


# ═══════════════════════════════════════════════════════════════
# 8. ATS SCORER v3
# ═══════════════════════════════════════════════════════════════
class ATSScorer:
    """
    5-component weighted ATS score (v3 weights):
      Skills Match          40%  unchanged — primary filter
      Experience Relevance  25%  unchanged — years + verbs + quant
      Projects/Achievements 15%  RAISED from 10% — evidence > degree
      Keyword Optimisation  15%  unchanged — TF-IDF + vocabulary
      Education Match        5%  LOWERED from 10% — rarely decisive alone

    Total = 100.
    Each component exposes: score, max, pct, why, improve, weight_label.
    """

    def __init__(self, sim_engine: SimilarityEngine):
        self._sim = sim_engine

    def score(self, resume_text: str, jd_text: str) -> dict:
        resume_skills, _ = SkillExtractor.extract(resume_text)
        sim       = self._sim.compute(resume_text, jd_text)
        sections  = SectionParser.detect_sections(resume_text)
        yrs       = SectionParser.extract_years(resume_text)
        edu_norm, degree = SectionParser.education_score(resume_text)
        action_ct = SectionParser.count_action_verbs(resume_text)
        has_quant = SectionParser.has_quantified_achievements(resume_text)
        has_proj, proj_count = SectionParser.detect_projects(resume_text)
        has_intern = SectionParser.has_internship(resume_text)
        online    = SectionParser.extract_online_presence(resume_text)
        contact   = ContactValidator.validate(resume_text)
        inferred  = ContextualSkillInferencer.infer(resume_text, resume_skills)
        exp_entries = SectionParser.extract_experience_entries(resume_text)

        resume_skills, skill_hier = SkillExtractor.extract(resume_text)
        jd_skills,    _           = SkillExtractor.extract(jd_text)

        certs, cert_count = CertificationAnalyser.extract(resume_text)
        domain_result     = ExperienceDomainAnalyser.analyse(
            resume_text, jd_text, resume_skills, jd_skills)
        cert_analysis = CertificationAnalyser.analyse_relevance(
            certs, cert_count, domain_result)
        patterns = PatternDetector.detect(
            resume_text, jd_text, yrs, len(resume_skills),
            domain_result, sections, cert_count, has_quant,
            has_proj, has_intern)

        components: dict = {}

        # ── 1. Skills Match — 40 pts ──────────────────────────────
        sk_pct      = sim["skill_score"]
        s1          = round(sk_pct * 0.40, 1)
        matched_n   = len(sim["matched_skills"])
        missing_n   = len(sim["missing_skills"])
        partial_n   = len(sim["partial_matches"])
        total_jd_sk = max(len(jd_skills), 1)

        if sk_pct >= 75:
            why1 = (f"{matched_n}/{total_jd_sk} required skills matched directly; "
                    f"{partial_n} partial match(es). Strong technical alignment.")
            fix1 = ("Excellent. Ensure matched skills appear inside experience bullets "
                    "with context — not just listed in a skills section.")
        elif sk_pct >= 45:
            top_m = sim["missing_skills"][:4]
            why1  = (f"{matched_n}/{total_jd_sk} required skills matched. "
                     f"{missing_n} missing, {partial_n} partially matched.")
            fix1  = ("Add missing skills: " + ", ".join(top_m) + ". "
                     "Integrate them into experience bullets, not just a skills list.")
        else:
            top_m = sim["missing_skills"][:5]
            why1  = (f"Only {matched_n}/{total_jd_sk} required skills matched. "
                     f"Major gaps: {missing_n} missing skills.")
            fix1  = ("Critical gaps: " + ", ".join(top_m) + ". "
                     "Reframe experience bullets to highlight any related work. "
                     "Add personal projects using these technologies.")
        components["Skills Match"] = {
            "score": s1, "max": 40, "pct": sk_pct,
            "icon": "🛠️", "why": why1, "improve": fix1,
            "weight_label": "40%",
            "detail": {
                "matched": sim["matched_skills"],
                "missing": sim["missing_skills"],
                "partial": sim["partial_matches"],
                "extra":   sim["extra_skills"],
            },
        }

        # ── 2. Experience Relevance — 25 pts ─────────────────────
        yrs_norm  = min(yrs / 8, 1.0) if yrs > 0 else 0.0
        verb_norm = min(action_ct / 15, 1.0)
        quant_b   = 0.15 if has_quant else 0.0
        dom_bonus = 0.10 if domain_result["domain_match_pct"] >= 66 else 0.0
        exp_pct   = round(min((yrs_norm*0.45 + verb_norm*0.30 + quant_b + dom_bonus), 1.0)*100, 1)
        s2        = round(exp_pct * 0.25, 1)

        if yrs >= 5:
            why2 = (f"{yrs}+ years of experience. {action_ct} action verbs detected. "
                    f"Quantified results: {'Yes' if has_quant else 'No'}. "
                    f"Domain relevance: {domain_result['relevance_label']}.")
            fix2 = ("Strong. Ensure every role shows: technology used → specific action → measurable outcome.")
        elif yrs >= 2:
            why2 = (f"{yrs} year(s) detected. {action_ct} action verbs. "
                    f"Quantified: {'Yes' if has_quant else 'No'}. "
                    f"Domain: {domain_result['relevance_label']}.")
            fix2 = ("Add quantified results to every bullet. "
                    "State impact: 'Reduced page load by 40%' not 'Improved performance'.")
        elif has_intern:
            why2 = (f"Internship experience detected. {action_ct} action verbs. No full-time work history.")
            fix2 = ("Add all internship details with dates, role title, and measurable output. "
                    "Include personal projects to compensate for limited experience.")
        else:
            why2 = (f"No work experience detected. {action_ct} action verbs found. "
                    "Fresher profile — scored on projects and skills instead.")
            fix2 = ("Add an Experience section even for academic projects. "
                    "Format: Role | Organisation | Date range | Bullet points with outcome.")
        components["Experience Relevance"] = {
            "score": s2, "max": 25, "pct": exp_pct,
            "icon": "💼", "why": why2, "improve": fix2,
            "weight_label": "25%",
            "detail": {
                "years": yrs,
                "action_verbs": action_ct,
                "has_quant": has_quant,
                "has_internship": has_intern,
                "domain_relevance": domain_result["relevance_label"],
            },
        }

        # ── 3. Projects & Achievements — 15 pts (raised from 10) ──
        proj_pct = 0.0
        if has_proj:       proj_pct += 40
        if proj_count >= 3: proj_pct += 15
        elif proj_count >= 1: proj_pct += 8
        if has_quant:      proj_pct += 25
        if cert_count >= 1: proj_pct += 10
        if ("achievements" in sections or "publications" in sections):
            proj_pct += 10
        proj_pct = min(proj_pct, 100)
        s3 = round(proj_pct * 0.15, 1)

        if proj_pct >= 80:
            why3 = (f"Strong evidence of concrete work. ~{proj_count} project(s) detected. "
                    f"Quantified results: {'Yes' if has_quant else 'No'}. "
                    f"Certifications: {cert_count}.")
            fix3 = "Ensure each project states: problem → tech used → your role → measurable outcome."
        elif proj_pct >= 40:
            why3 = (f"Some project evidence found (~{proj_count} detected). "
                    f"Quantified: {'Yes' if has_quant else 'No'}. Certs: {cert_count}.")
            fix3 = ("Add GitHub links, live demo URLs, or performance metrics to each project. "
                    "Aim for 3+ projects with concrete outcomes.")
        else:
            why3 = "Minimal evidence of projects, achievements, or certifications."
            fix3 = ("Add a Projects section: project name | tech stack | your contribution | outcome. "
                    "Even academic or personal side projects count significantly.")
        components["Projects & Achievements"] = {
            "score": s3, "max": 15, "pct": proj_pct,
            "icon": "🚀", "why": why3, "improve": fix3,
            "weight_label": "15%",
            "detail": {
                "has_projects": has_proj,
                "project_count": proj_count,
                "has_quant": has_quant,
                "cert_count": cert_count,
                "github": online.get("github", False),
            },
        }

        # ── 4. Keyword Optimisation — 15 pts ──────────────────────
        kw_pct    = sim["keyword_score"]
        tf_pct    = sim["tfidf_score"]
        comb_kw   = round(kw_pct * 0.6 + tf_pct * 0.4, 1)
        s4        = round(comb_kw * 0.15, 1)

        if comb_kw >= 65:
            why4 = (f"Good vocabulary alignment. TF-IDF similarity: {tf_pct}%, "
                    f"keyword overlap: {kw_pct}%.")
            fix4 = "Mirror specific JD phrases in experience bullets for even tighter alignment."
        elif comb_kw >= 40:
            why4 = (f"Moderate alignment. TF-IDF: {tf_pct}%, overlap: {kw_pct}%. "
                    "Some JD terminology is absent.")
            fix4 = ("Use the JD's exact phrasing. If JD says 'RESTful microservices', "
                    "use that phrase — not 'API development'.")
        else:
            why4 = (f"Low keyword alignment. TF-IDF: {tf_pct}%, overlap: {kw_pct}%. "
                    "Resume vocabulary doesn't match the JD.")
            fix4 = ("Read the JD carefully and mirror its exact terms throughout. "
                    "ATS scanners match exact phrases, not semantic equivalents.")
        components["Keyword Optimisation"] = {
            "score": s4, "max": 15, "pct": comb_kw,
            "icon": "🔑", "why": why4, "improve": fix4,
            "weight_label": "15%",
        }

        # ── 5. Education Match — 5 pts (lowered from 10) ──────────
        jd_edu_req = _detect_jd_education_requirement(jd_text)
        edu_pct    = round(edu_norm * 100, 1)
        if jd_edu_req and edu_norm < 0.4:
            edu_pct = max(edu_pct - 10, 0)
        if cert_count >= 2:
            edu_pct = min(edu_pct + 20, 100)  # Strong certs compensate
        s5 = round(edu_pct * 0.05, 1)

        if edu_norm >= 0.8:
            why5 = f"High education level detected ({degree})."
            fix5 = "Ensure graduation year, institution, and specialisation are clearly stated."
        elif edu_norm >= 0.5:
            why5 = f"Relevant degree detected ({degree})."
            fix5 = "Add degree name, institution, year, and CGPA/percentage for clarity."
        elif edu_norm >= 0.2:
            why5 = f"Basic education detected ({degree}). {cert_count} cert(s) partially compensate."
            fix5 = "Strengthen with targeted certifications. State your qualification fully."
        else:
            why5 = "Education section weak or not detected."
            fix5 = "Add: Degree, Institution, Year, Specialisation, CGPA/Percentage."
        components["Education Match"] = {
            "score": s5, "max": 5, "pct": edu_pct,
            "icon": "🎓", "why": why5, "improve": fix5,
            "weight_label": "5%",
        }

        # ── Final score ──
        final = round(min(sum(c["score"] for c in components.values()), 98.0), 1)

        # ── Recruiter verdict (explicit YES / MAYBE / NO) ──
        verdict_data = _compute_recruiter_verdict(
            final, sim["overall"], domain_result, cert_analysis,
            patterns, yrs, cert_count, len(resume_skills),
            has_quant, sections)

        return {
            "final_score":      final,
            "components":       components,
            "similarity":       sim,
            "sections":         sections,
            "years_exp":        yrs,
            "degree":           degree,
            "certifications":   certs,
            "cert_count":       cert_count,
            "cert_analysis":    cert_analysis,
            "skill_hier":       skill_hier,
            "resume_skills":    resume_skills,
            "jd_skills":        jd_skills,
            "action_verbs":     action_ct,
            "has_quant":        has_quant,
            "has_projects":     has_proj,
            "project_count":    proj_count,
            "has_internship":   has_intern,
            "online_presence":  online,
            "exp_entries":      exp_entries,
            "domain_analysis":  domain_result,
            "patterns":         patterns,
            "recruiter_verdict": verdict_data,
            "contact_info":    contact,
            "inferred_skills": inferred,
        }


# ═══════════════════════════════════════════════════════════════
# 9. RECRUITER VERDICT ENGINE  (NEW in v3)
# ═══════════════════════════════════════════════════════════════
def _compute_recruiter_verdict(
    ats_score: float,
    jd_match: float,
    domain_result: dict,
    cert_analysis: dict,
    patterns: list[dict],
    yrs: int,
    cert_count: int,
    skill_count: int,
    has_quant: bool,
    sections: list[str],
) -> dict:
    """
    Produces an explicit recruiter-grade verdict:
      decision:    "YES" | "MAYBE" | "NO"
      confidence:  "High" | "Medium" | "Low"
      short_reason: 1-line recruiter headline
      reasoning:   list of bullet-point reasons
      next_action: what the recruiter should do next
    """
    reasons: list[str] = []
    score_signals: list[int] = []  # +1 positive, -1 negative, 0 neutral

    # ATS score signal
    if ats_score >= 75:
        score_signals.append(1)
        reasons.append(f"Strong ATS score ({ats_score}/100)")
    elif ats_score >= 55:
        score_signals.append(0)
        reasons.append(f"Moderate ATS score ({ats_score}/100) — some gaps remain")
    else:
        score_signals.append(-1)
        reasons.append(f"Weak ATS score ({ats_score}/100) — significant gaps")

    # JD match signal
    if jd_match >= 60:
        score_signals.append(1)
        reasons.append(f"JD match is strong ({jd_match}%)")
    elif jd_match >= 35:
        score_signals.append(0)
        reasons.append(f"JD match is moderate ({jd_match}%)")
    else:
        score_signals.append(-1)
        reasons.append(f"Low JD match ({jd_match}%) — skill vocabulary doesn't align well")

    # Domain signal
    dom_pct = domain_result.get("domain_match_pct", 0)
    if dom_pct >= 66:
        score_signals.append(1)
        reasons.append(f"Experience domain matches JD ({domain_result['relevance_label']})")
    elif dom_pct >= 33:
        score_signals.append(0)
        reasons.append(f"Partial domain match — {domain_result['insight'][:80]}")
    else:
        score_signals.append(-1)
        reasons.append(f"Domain mismatch — {domain_result['insight'][:80]}")

    # Experience signal
    if yrs >= 4:
        score_signals.append(1)
        reasons.append(f"{yrs}+ years of relevant experience")
    elif yrs >= 1:
        score_signals.append(0)
        reasons.append(f"{yrs} year(s) of experience — entry to mid level")
    else:
        score_signals.append(-1)
        reasons.append("No work experience detected (fresher profile)")

    # Certs signal
    if cert_count >= 2:
        score_signals.append(1)
        reasons.append(f"{cert_count} certifications — well credentialed")
    elif cert_count == 1:
        score_signals.append(0)
        reasons.append("1 certification found")

    # Quantification
    if has_quant:
        score_signals.append(1)
        reasons.append("Resume contains quantified achievements — strong impact evidence")
    elif yrs >= 2:
        score_signals.append(-1)
        reasons.append("No quantified results despite experience — passive language pattern")

    # Pattern penalties
    for pat in patterns:
        if pat["severity"] == "warn":
            score_signals.append(-1)
            reasons.append(pat["label"] + " — " + pat["description"][:80])

    # Critical sections
    missing_crit = [s for s in ["experience","education","skills"] if s not in sections]
    if missing_crit:
        score_signals.append(-2)
        reasons.append("Critical sections missing: " + ", ".join(s.upper() for s in missing_crit))

    # Tally
    total = sum(score_signals)
    pos   = sum(1 for s in score_signals if s == 1)
    neg   = sum(1 for s in score_signals if s < 0)

    # Decision
    if total >= 3 and neg <= 1:
        decision = "YES"
        dec_color = "#22c55e"
        dec_bg    = "#0a2a1a"
        short_reason = _build_short_reason(ats_score, jd_match, yrs, domain_result)
        next_action  = "Proceed to phone screen. Focus interview on projects and impact metrics."
        confidence   = "High" if total >= 5 else "Medium"
    elif total >= 0 and neg <= 3:
        decision = "MAYBE"
        dec_color = "#f0c040"
        dec_bg    = "#2a2a0a"
        short_reason = _build_short_reason(ats_score, jd_match, yrs, domain_result)
        next_action  = ("Consider a brief (15 min) screening call to clarify skill gaps. "
                        "Ask about: " + ", ".join(
                            [r[:40] for r in reasons if "-" in r.lower() or "missing" in r.lower()])[:80] + ".")
        confidence   = "Medium" if total >= 1 else "Low"
    else:
        decision = "NO"
        dec_color = "#ef4444"
        dec_bg    = "#2a0a0a"
        short_reason = _build_short_reason(ats_score, jd_match, yrs, domain_result)
        next_action  = "Does not meet minimum requirements. Keep profile on file for future entry-level openings."
        confidence   = "High" if total <= -4 else "Medium"

    return {
        "decision":     decision,
        "dec_color":    dec_color,
        "dec_bg":       dec_bg,
        "confidence":   confidence,
        "short_reason": short_reason,
        "reasoning":    reasons,
        "next_action":  next_action,
        "pos_signals":  pos,
        "neg_signals":  neg,
    }


def _build_short_reason(ats: float, jd: float, yrs: int, domain: dict) -> str:
    """One-line recruiter headline."""
    dom_str = domain.get("relevance_label","")
    if ats >= 75 and jd >= 60:
        return f"Strong technical match ({dom_str.lower()}). {yrs}+ yr(s) experience."
    elif ats >= 55 and jd >= 35:
        return f"Decent fit with addressable gaps ({dom_str.lower()}). {yrs}+ yr(s) experience."
    elif ats >= 35:
        return f"Partial match — skill and keyword gaps need attention. {dom_str}."
    else:
        return f"Insufficient match for this role. Consider rejecting or redirecting."


# ═══════════════════════════════════════════════════════════════
# 10. RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════════════
class RecommendationEngine:
    """
    Generates prioritised, actionable improvements.
    Priority order: critical → high → medium → low.
    """

    @staticmethod
    def generate(scoring_result: dict, jd_text: str) -> list[dict]:
        recs: list[dict] = []
        sim     = scoring_result["similarity"]
        comps   = scoring_result["components"]
        secs    = scoring_result["sections"]
        yrs     = scoring_result["years_exp"]
        quant   = scoring_result["has_quant"]
        cert_ct = scoring_result["cert_count"]
        missing = sim["missing_skills"]
        partial = sim["partial_matches"]
        domain  = scoring_result["domain_analysis"]
        online  = scoring_result["online_presence"]
        patterns= scoring_result["patterns"]

        def rec(priority, category, title, action, impact):
            recs.append({"priority": priority, "category": category,
                         "title": title, "action": action, "impact": impact})

        # Critical: missing required skills
        if missing:
            rec("critical", "Skills Match",
                f"Add {len(missing)} missing JD skills",
                ("JD requires these skills that aren't in your resume: "
                 + ", ".join(missing[:8]) + ". "
                 "Add each with context in experience bullets or projects."),
                "Up to +20 pts")

        # Critical: missing critical sections
        for must in ["experience","education","skills"]:
            if must not in secs:
                rec("critical", "Section Structure",
                    f"Add {must.title()} section",
                    (f"'{must.upper()}' section header not detected. "
                     "ATS systems will auto-reject resumes missing these three sections."),
                    "Critical — affects all scores")

        # High: no quantified achievements
        if not quant and yrs >= 1:
            rec("high", "Experience Relevance",
                "Quantify your achievements",
                ("No numbers found in experience bullets. Add metrics to every achievement: "
                 "'Reduced API latency by 35%', 'Deployed to 50k+ users', 'Managed team of 6'. "
                 "Quantification is the single highest-impact resume change."),
                "Up to +8 pts")

        # High: partial skill upgrades
        if partial:
            acts = [f"'{r}' → add '{j}' explicitly" for r, j in partial[:4]]
            rec("high", "Skills Match",
                f"Upgrade {len(partial)} partial skill match(es)",
                "You have the parent skill but JD wants the specific sub-skill. "
                + " | ".join(acts) + ".",
                "Up to +6 pts")

        # High: domain mismatch
        if domain.get("relevance_label") == "Domain Mismatch":
            rec("high", "Experience Relevance",
                "Reframe experience for domain alignment",
                domain["insight"] + " Rewrite 2-3 experience bullets to highlight "
                "transferable skills relevant to: " +
                ", ".join(domain.get("jd_domains",["the JD domain"])[:2]) + ".",
                "Up to +10 pts")

        # High: no certifications
        if cert_ct == 0:
            rec("high", "Projects & Achievements",
                "Add at least one relevant certification",
                ("No certifications detected. Certifications add credibility especially "
                 "for domains like Cloud, Data, PM, or Finance. "
                 "Even a free Google/Coursera cert demonstrates commitment."),
                "Up to +5 pts")

        # Medium: no summary
        if "summary" not in secs:
            rec("medium", "Keyword Optimisation",
                "Add a professional summary",
                ("A 2-3 line summary at the top is the first thing recruiters read. "
                 "Include: your title, years of experience, and top 2-3 skills from the JD. "
                 "Example: 'Python ML Engineer with 3 years building NLP pipelines at scale.'"),
                "Improves first impression + keyword density")

        # Medium: no projects
        if not scoring_result["has_projects"] and "projects" not in secs:
            rec("medium", "Projects & Achievements",
                "Add a Projects section",
                ("No projects detected. Add 2-3 projects showing: name, tech stack, your role, and measurable outcome. E.g. Built fraud detection pipeline using Spark, reducing false positives by 22%."),
                "Up to +6 pts")

        # Medium: no GitHub for tech roles
        jd_t = jd_text.lower()
        is_tech = any(w in jd_t for w in ["engineer","developer","data","ml","software","backend"])
        if is_tech and not online.get("github", False):
            rec("medium", "Projects & Achievements",
                "Add GitHub / portfolio link",
                ("No GitHub profile detected. For technical roles, a GitHub link with "
                 "active repositories is expected by most hiring managers and significantly "
                 "increases interview conversion rates."),
                "Strong qualitative signal")

        # Medium: low keyword alignment
        if comps["Keyword Optimisation"]["pct"] < 45:
            rec("medium", "Keyword Optimisation",
                "Mirror JD language throughout",
                ("Resume vocabulary diverges significantly from JD. "
                 "Read the JD and use its exact phrases in bullets. "
                 "If JD says 'distributed systems', don't write 'large-scale architecture'. "
                 "ATS matches exact phrases, not synonyms."),
                "Up to +5 pts")

        # Medium: experience duration not stated
        if yrs == 0 and "experience" in secs:
            rec("medium", "Experience Relevance",
                "State experience duration explicitly",
                ("Experience section found but no dates or duration detected. "
                 "Always include: 'Role Title — Company | Jan 2022 – Dec 2023'. "
                 "ATS systems use dates to calculate years of experience."),
                "Critical for experience scoring")

        # Low: education details thin
        if comps["Education Match"]["pct"] < 40:
            rec("low", "Education Match",
                "Expand education details",
                ("Add full education entry: Degree name (B.Tech/M.Sc/MBA), "
                 "Institution name, Graduation year, CGPA/percentage. "
                 "Include relevant coursework if a fresher."),
                "Up to +2 pts")

        # Low: weak action verbs
        if scoring_result["action_verbs"] < 5:
            rec("low", "Experience Relevance",
                "Use stronger action verbs",
                ("Replace passive language: "
                 "'Responsible for backend' → 'Architected Python backend serving 2M+ users'. "
                 "Strong verbs: Led, Built, Deployed, Optimised, Reduced, Scaled, Delivered."),
                "Improves readability and verb scoring")

        # Pattern-based recs
        for pat in patterns:
            if pat["severity"] == "warn" and pat["label"] == "Lacks Quantified Results":
                if not quant:
                    rec("high", "Experience Relevance",
                        "Add metrics to every experience bullet",
                        pat["description"] + " Examples: %, $, # of users, time saved.",
                        "High impact on experience score")

        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recs.sort(key=lambda r: priority_order.get(r["priority"], 4))
        return recs


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def _detect_jd_education_requirement(jd_text: str) -> Optional[str]:
    if not isinstance(jd_text, str): return None
    t = jd_text.lower()
    for kw in ["phd","master","mba","m.tech","bachelor","b.tech","b.e","degree","diploma"]:
        if kw in t:
            return kw.upper()
    return None


def parse_upload(uploaded_file) -> str:
    raw = uploaded_file.read()
    fn  = uploaded_file.name.lower()
    try:
        if fn.endswith(".pdf"):
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(raw))
            return "\n".join(p.extract_text() or "" for p in reader.pages).strip()
        elif fn.endswith(".docx"):
            from docx import Document
            doc = Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs).strip()
        else:
            return raw.decode("utf-8", errors="ignore")
    except Exception as e:
        return f"Parse error: {e}"