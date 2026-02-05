from sqlalchemy import create_engine, Column, Integer, String, Boolean, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# SQLite database URL
SQLALCHEMY_DATABASE_URL = "sqlite:///./neurosearch.db"

# Create engine
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# --- Models ---

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

    # Relationship to projects
    projects = relationship("Project", back_populates="owner")


class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, index=True)
    name = Column(String)
    train_path = Column(String)
    index_file = Column(String)
    metadata_file = Column(String)
    created_at = Column(Float)
    is_default = Column(Boolean, default=False)
    
    # Foreign key to User
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationship to User
    owner = relationship("User", back_populates="projects")

# Create tables
def init_db():
    Base.metadata.create_all(bind=engine)
