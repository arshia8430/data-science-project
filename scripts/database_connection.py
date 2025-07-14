import os
import sqlalchemy

def get_db_engine():
    """
    Dynamically creates and returns a SQLAlchemy engine connected to the project's SQLite database.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    project_root = os.path.dirname(script_dir)
    
    db_path = os.path.join(project_root, 'database', 'dataset.db')

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found at: {db_path}. Please run the import script first.")
        

    engine = sqlalchemy.create_engine(f'sqlite:///{db_path}')
    
    return engine

if __name__ == '__main__':

    print("Attempting to connect to the database...")
    try:
        engine = get_db_engine()
        with engine.connect() as connection:
            print("✅ Database connection successful!")
            print(f"Engine Dialect: {connection.dialect.name}")
    except Exception as e:
        print(f"❌ Database connection failed. Error: {e}")