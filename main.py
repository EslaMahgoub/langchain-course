import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():
    information = """
    Dario Amodei (born 1983) is an American artificial intelligence (AI) researcher and entrepreneur. He is the co-founder and CEO of Anthropic, the company behind the large language model series Claude.[1] He was previously the vice president of research at OpenAI.[2][3]

    In his capacity as Anthropic's CEO, he often writes on the benefits and risks of advanced AI systems.[4] He is a proponent of an "entente" strategy in which a coalition of democratic nations use advanced AI systems in military applications to achieve a decisive advantage over adversaries, while sharing the benefits with cooperating nations.[5][6][7]

    Early life
    Dario Amodei was born in San Francisco, California, in 1983.[8] His sister, Daniela, was born four years later.[8] His father, Riccardo Amodei, an Italian-American leather craftsman from Massa Marittima, Tuscany, died when Amodei was a young adult.[9][10] His mother is Elena Engel, a Jewish American born in Chicago who worked as a project manager for libraries.[8]

    Education
    Dario grew up in San Francisco and graduated from Lowell High School.[11] He was a member of the US Physics Olympiad team in 2000.[12] Amodei began college at Caltech, where he worked with Tom Tombrello as one of his Physics 11 students. He later transferred to Stanford University, where he received his bachelor's degree in physics.[13] He also holds a PhD in biophysics from Princeton University, where he studied electrophysiology of neural circuits.[14] He was a postdoctoral scholar at the Stanford University School of Medicine.[15]

    Career
    From November 2014 until October 2015, he worked at Baidu.[16] After that, he worked at Google.[17] In 2016, Amodei joined OpenAI.[18]

    In 2021, Dario and his sister, Daniela, founded Anthropic along with other former senior members of OpenAI.[19][20] The Amodei siblings were among those who left OpenAI due to directional differences.[21] As of February 2026, Anthropic has an estimated value of $380 billion,[22] with Forbes estimating Amodei's net worth to be $7 billion.[23]

    In November 2023, the board of directors of OpenAI approached Amodei about replacing Sam Altman and potentially merging the two startups. Amodei declined both offers.[24]

    In 2025, Time magazine listed Amodei as one of the world's 100 most influential people.[25] He was also named as one of the "Architects of AI" for Time's Person of the Year.[26]

    Views
    Amodei has written about both the potential benefits and the risks of advanced AI systems.

    Benefits of AI
    In October 2024, Amodei published an essay titled "Machines of Loving Grace", speculating about how AI could improve human welfare.[6] In it, he writes, "I think that most people are underestimating just how radical the upside of AI could be, just as I think most people are underestimating how bad the risks could be."[5] The essay described a vision of civilization where the risks of AI had been addressed and powerful AI was applied to raise the quality of life for everyone, suggesting that AI could contribute to enormous advances in biology, neuroscience, economic development, global peace, and work and meaning.[5]

    In the article, Amodei also stresses the importance "that democracies have the upper hand on the world stage when powerful AI is created", and argues for an "entente" strategy where a coalition of democracies use AI to achieve a decisive strategic and military advantage over their adversaries, while distributing the benefits to all cooperating democratic nations.[5]

    Risks of AI
    In January 2026, Amodei published a follow-up essay titled "The Adolescence of Technology", which focuses on the risks posed by powerful AI[27][28][29] and expands on his earlier statements about these risks.[30][31][4][32][33] In the essay, Amodei identifies five major categories of AI risk.

    The first category concerns the possibility that AI systems develop goals or behaviors misaligned with human intentions. He notes that such behaviors have already been observed in testing at Anthropic, including AI models engaging in deception, blackmail, and scheming.[27][28]

    The second category involves misuse of AI for destruction by individuals or small groups, with Amodei expressing particular concern about biological weapons. He warns that AI could enable people without specialized training to create weapons of mass destruction.[27][28]

    The third category concerns misuse of AI by powerful actors to seize or maintain power. Amodei cautions that AI could enable authoritarian governments to conduct unprecedented surveillance, deploy autonomous weapons, and engage in mass propaganda. He identifies the Chinese Communist Party as the greatest threat in this regard, arguing that democracies must maintain AI leadership to prevent a "global totalitarian dictatorship."[27][34]

    The fourth category addresses economic disruption, including mass labor displacement and concentration of wealth. Amodei notes that AI could displace half of all entry-level white-collar jobs within one to five years, and warns of wealth concentration exceeding that of the Gilded Age, with personal fortunes potentially reaching into the trillions of dollars.[27][35][29][28]

    The fifth category encompasses indirect effects and unknown unknowns, including rapid advances in biology that could alter human lifespans or human intelligence, unhealthy changes to human life from AI interaction, and challenges to human purpose in a world where AI exceeds human capabilities across virtually all domains.[27]
    """

    summary_template = """
        given the information {information} about a person I want you to create:
        1. A short Summary
        2. Two interesting facts about him
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    LLM_CONFIG = {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key": os.environ.get("OPENROUTER_API_KEY"),
        "model": "gpt-5-nano",
        "temperature": 0,
    }

    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    openai_llm = ChatOpenAI(**LLM_CONFIG)
    parser = StrOutputParser()
    ollama = ChatOllama(temperature=0, model="gemma3:4b")

    chain = summary_prompt_template | ollama | parser
    response = chain.invoke(input={"information": information})

    print(response)

if __name__ == "__main__":
    main()
