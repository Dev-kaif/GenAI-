from langchain_community.document_loaders import SeleniumURLLoader

urls = [
   "https://docs.chaicode.com/youtube/getting-started/",
]

loader = SeleniumURLLoader(urls=urls)

docs = loader.load()

print(docs)