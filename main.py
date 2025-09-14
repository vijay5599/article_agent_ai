from fastapi import FastAPI, HTTPException
from openai import OpenAI
import json
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
import os

load_dotenv()
app = FastAPI()
client = OpenAI()


class ArticleBase(BaseModel):
    title: Optional[str] = None


class ArticleCreate(ArticleBase):
    topic: Optional[str] = None
    inputs: Optional[str] = (
        None  # JSON string: {industry, country, featured, tone_of_strength, humanize}
    )
    instructions: Optional[str] = None
    is_active: Optional[bool] = None
    content: Optional[str] = ""
    edited_content: Optional[str] = ""
    state: Optional[int] = None  # {draft, published}


@app.get("/")
def root():
    return {"status": "ok", "openai_key_loaded": bool(os.getenv("OPENAI_API_KEY"))}


@app.post("/generate-article")
async def generate_article(article: ArticleCreate):
    try:  # Parse extra inputs
        extras = {}
        if article.inputs:
            try:
                extras = json.loads(article.inputs)
            except:
                pass  # Build prompt
            prompt = f""" Write an article based on: Title: {article.title or article.topic} Topic: {article.topic} Instructions: {article.instructions} Inputs: {extras} Requirements: - Professional tone ({extras.get('tone_of_strength', 'neutral')}) - Humanized: {extras.get('humanize', True)} - Must be engaging and clear. """
            # Call AI model
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            generated_content = response.choices[0].message.content
            return {
                "title": article.title or article.topic,
                "content": generated_content,
                "state": article.state or 0,
                "is_active": (
                    article.is_active if article.is_active is not None else True
                ),
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
