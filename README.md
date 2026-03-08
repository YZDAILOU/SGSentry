## Hackomania 2026 Team Beyond Kinetic Project SGSentry AI ##

Fact Checking AI (Video & Image Input)

This project performs fact checking based on video or image input.
It extracts information from media files and analyzes the credibility of the content using AI models.

Project Setup
1. Clone the Repository
git clone https://github.com/YZDAILOU/SGSentry.git
cd SGSentry
2. Create Python Virtual Environment
python -m venv venv

Activate the virtual environment:

Mac / Linux
source venv/bin/activate
Windows
venv\Scripts\activate

3. Install Dependencies

Install all required packages using:

pip install -r requirements.txt

4. Setup Environment Variables

Create a .env file in the root folder and add the required API keys.

Example:

# Due to security reason, can't share the keys with you :)
OPEN_PAGERANK_KEY=
GOOGLE_FACT_CHECK_KEY=
GOOGLE_API_KEY=
CH_HOST=
CH_PORT=
CH_USER=
CH_PASS=
CH_MCP_SERVER_TRANSPORT=
OPENAI_API_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_PUBLIC_KEY=
LANGFUSE_BASE_URL=

5. Run the Project

Run the main application:

python main.py