import requests
from transformers import pipeline, BartTokenizer
import base64
import pandas as pd
import os

GITHUB_TOKEN = "YOUR_GITHUB_TOKEN_HERE"
SUMMARY_MODEL = "facebook/bart-large-cnn"
SUMMARY_REVISION = "main"
API_HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}

tokenizer = BartTokenizer.from_pretrained(SUMMARY_MODEL)
summarizer = pipeline("summarization", model=SUMMARY_MODEL, tokenizer=tokenizer, revision=SUMMARY_REVISION)


def get_repo_info(repo_url):
    repo_api_url = f"https://api.github.com/repos/{'/'.join(repo_url.split('/')[-2:])}"
    response = requests.get(repo_api_url, headers=API_HEADERS)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Error {response.status_code} while accessing the repository.")
        return None


def get_commit_count(repo_url):
    commits_url = f"https://api.github.com/repos/{'/'.join(repo_url.split('/')[-2:])}/commits"
    response = requests.get(commits_url, headers=API_HEADERS)

    if response.status_code == 200:
        return len(response.json())
    else:
        print(f"❌ Error {response.status_code} while accessing commits.")
        return 0


def get_contributors(repo_url):
    contributors_url = f"https://api.github.com/repos/{'/'.join(repo_url.split('/')[-2:])}/contributors"
    response = requests.get(contributors_url, headers=API_HEADERS)

    if response.status_code == 200:
        return [contributor["login"] for contributor in response.json()][:5]
    else:
        print(f"❌ Error {response.status_code} while accessing contributors.")
        return []


def get_readme_summary(repo_url):
    readme_url = f"https://api.github.com/repos/{'/'.join(repo_url.split('/')[-2:])}/contents/README.md"
    response = requests.get(readme_url, headers=API_HEADERS)

    print(f"🔍 Fetching README from: {readme_url}")
    print(f"✅ Response Code: {response.status_code}")

    if response.status_code == 200:
        readme_content = response.json()
        readme_text = base64.b64decode(readme_content["content"]).decode('utf-8')

        if len(readme_text) > 1024:  # Adjust according to your needs
            readme_text = readme_text[:1024]

        inputs = tokenizer(readme_text, return_tensors='pt', max_length=1024, truncation=True)

        try:
            summarized_text = summarizer(readme_text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
            return summarized_text
        except Exception as e:
            print(f"❌ Error during summarization: {e}")
            return "Summarization failed."
    else:
        print(f"❌ Error {response.status_code} while accessing the README.")

        try:
            print("Response content:", response.json())
        except Exception:
            print("Response content could not be parsed.")
        return "No summary available."


def analyze_github_repo(repo_url):
    if not isinstance(repo_url, str) or not repo_url.startswith("https://github.com/"):
        print(f"❌ Invalid URL: {repo_url}. It should be a valid GitHub URL.")
        return None

    repo_data = get_repo_info(repo_url)

    if not repo_data:
        return "Unable to retrieve repository information."

    repo_name = repo_data.get("name", "Name not available")
    description = repo_data.get("description", "Description not available")
    owner = repo_data.get("owner", {}).get("login", "Owner unknown")
    stars = repo_data.get("stargazers_count", 0)
    forks = repo_data.get("forks_count", 0)
    open_issues = repo_data.get("open_issues_count", 0)

    commit_count = get_commit_count(repo_url)

    contributors = get_contributors(repo_url)

    readme_summary = get_readme_summary(repo_url)

    report = {
        "repository_name": repo_name,
        "owner": owner,
        "description": description,
        "stars": stars,
        "forks": forks,
        "commit_count": commit_count,
        "open_issues": open_issues,
        "top_contributors": ', '.join(contributors) if contributors else "No contributors found",
        "readme_summary": readme_summary,
        "url": repo_url
    }

    return report


input_file = 'urls.xlsx'  # Path to your Excel file
output_csv_file = 'CSV/results.csv'  # Path to output CSV file

csv_directory = os.path.dirname(output_csv_file)
if not os.path.exists(csv_directory):
    os.makedirs(csv_directory)

if not os.path.exists(output_csv_file):
    pd.DataFrame(columns=["repository_name", "owner", "description", "stars", "forks", "commit_count", "open_issues", "top_contributors", "readme_summary", "url"]).to_csv(output_csv_file, index=False, encoding='utf-8-sig')

xls = pd.ExcelFile(input_file)
df = pd.read_excel(xls, sheet_name='urls')  # Adjust sheet name if necessary

total_repositories = len(df)

for index, row in df.iterrows():
    repo_url = row['url']  # Adjust column name if necessary
    report = analyze_github_repo(repo_url)

    if report:  # Only append if the report is not None
        report_df = pd.DataFrame([report])

        report_df.to_csv(output_csv_file, mode='a', index=False, header=False, encoding='utf-8-sig')

        print("\n" + "="*40)
        print(f"Repository Name: {report['repository_name']}")
        print(f"Owner: {report['owner']}")
        print(f"Description: {report['description']}")
        print(f"Stars: {report['stars']}")
        print(f"Forks: {report['forks']}")
        print(f"Commit Count: {report['commit_count']}")
        print(f"Open Issues: {report['open_issues']}")
        print(f"Top Contributors: {report['top_contributors']}")
        print(f"README Summary: {report['readme_summary']}")
        print(f"Repository URL: {report['url']}")
        print("="*40)
        print()

print(f"\nTotal Repositories Processed: {total_repositories}")
print(f"Reports saved to {output_csv_file}.")