import requests
import pandas as pd
import time
import sqlalchemy as sa
import os


# Create a connection to the ALeRCE database
def create_conn_alerce(access_time: int = 2):
  """
  Create a connection to the ALeRCE database.
  
  Args:
    access_time: Integer representing the time limit for the connection.
    2 for default access, 10 for extended access.
    
  Returns:
    SQLAlchemy engine object
  
  Raises:
    ValueError: If URL fetch fails or credentials are invalid
  """
  # Common URL for both access levels
  url = "https://raw.githubusercontent.com/alercebroker/usecases/master/alercereaduser_v4.json"
  
  # For security, get extended access credentials from environment variables
  user_extended = os.environ.get('ALERCE_USER_EXTENDED', '')
  pass_extended = os.environ.get('ALERCE_PASS_EXTENDED', '')
  
  n_tries = 3
  params = None
  params = {
          "dbname" : "ztf",
          "user" : "alerceread",
          "host": "54.205.99.47",
          "password" : "w*C*u8AXZ4e%d+zv"
      }
  
  if params is None:
    # Fetch parameters from URL with retry logic
    for n_try in range(1, n_tries + 1):
      try:
        response = requests.get(url)
        if response.status_code != 200:
          if n_try < n_tries:
            time.sleep(2 ** n_try)  # exponential backoff
            continue
          else:
            raise ValueError(f"Failed to fetch URL: {url}, Status Code: {response.status_code}")
        
        params = response.json().get('params')
        if not params:
          raise ValueError("Missing 'params' in the JSON response")
        break
          
      except requests.RequestException as e:
        if n_try < n_tries:
          time.sleep(2 ** n_try)
          continue
        else:
          raise ValueError(f"Network error when fetching {url}: {str(e)}")
      except ValueError as e:
        if "JSON" in str(e):
          raise ValueError("Invalid JSON response from URL")
        else:
          raise e
  
  # Create connection string based on access level
  if access_time == 2:
    conn_string = f"postgresql+psycopg2://{params['user']}:{params['password']}@{params['host']}/{params['dbname']}"
  elif access_time == 10:
    if not user_extended or not pass_extended:
      raise ValueError("Extended access credentials not found in environment variables")
    conn_string = f"postgresql+psycopg2://{user_extended}:{pass_extended}@{params['host']}/{params['dbname']}"
  else:
    raise ValueError(f'Access time {access_time} not supported')
  
  # Create and return engine with connection pooling disabled
  engine = sa.create_engine(conn_string, poolclass=sa.pool.NullPool)
  return engine


def run_sql_alerce(
    sql: str, 
    access_time: int = 2, 
    n_tries: int = 3, 
    query_time: bool = False
):
  ''' Execute the SQL query at the ALeRCE database and return the result
    Args:
        sql: SQL query to execute
        access_time: Integer representing the time limit for the connection. 
        2 for default access, 10 for extended access.
        n_tries: Number of tries to execute the query
        query_time: Boolean indicating whether to track query execution time
    Returns:
    query: pandas dataframe with the result of the query
    error: error message if the query could not be executed
    execution_time: time taken to execute the query    
  '''

  try:
    engine = create_conn_alerce(access_time=access_time)
  except ValueError as e:
    print(f"Error creating connection: {str(e)}")
    return None, e
    
  query = None
  error = None
  execution_time = None
  
  try:
    for n_try in range(n_tries):
      try:
        with engine.begin() as conn:
          start_time = time.time()
          query = pd.read_sql_query(sa.text(sql), conn)
          if query_time:
            execution_time = time.time() - start_time
          error = None
          break
      except Exception as e:
        error = e
        if n_try < n_tries - 1:
          time.sleep(2 * n_try)  # 
        else:
          # Last attempt failed, keep the error
          pass
  
  finally:
    # Always dispose of the engine to close connections
    engine.dispose()

  if query_time:
    return query, error, execution_time
  else:
    return query, error
