import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('db/idea-hub.db')
cursor = conn.cursor()

# Create a table with additional fields
cursor.execute('''
    CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY,
        topic TEXT NOT NULL,
        difficulty TEXT NOT NULL,
        project_title TEXT NOT NULL,
        description TEXT NOT NULL,
        youtube_link TEXT,
        github_link TEXT
    )
''')

# Sample data insertion
cursor.executemany('''
    INSERT INTO projects (topic, difficulty, project_title, description, youtube_link, github_link)
    VALUES (?, ?, ?, ?, ?, ?)
''', [
    ("Web Development", "Intermediate", "Portfolio Website",
     "A personal portfolio website showcasing skills and projects.",
     "https://youtube.com/sample_portfolio",
     "https://github.com/sample/portfolio"),
    ("Machine Learning", "Advanced", "Sentiment Analysis",
     "A model to classify sentiments in user reviews.",
     "https://youtube.com/sample_ml_project",
     "https://github.com/sample/ml_project")
])

conn.commit()
conn.close()
