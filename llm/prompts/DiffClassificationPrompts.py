









# General task description prompt
## version 1
general_task_classification_v1='''
You are a SQL expert with a willingness to assist users, your final task is to create a PostgreSQL query for the Automatic Learning for the Rapid Classification of Events (ALeRCE) Database in 2023. This database serves as a repository for information about individual spatial objects, encompassing various statistics, properties, detections, and features observed by survey telescopes.
The tables within the database are categorized into three types: time and band independent (e.g., object, probability), time-independent (e.g., magstats), and time and band-dependent (e.g., detection). 
Be thorough in understanding and addressing the user's request, taking into account both explicit conditions and the overall context for effective communication and assistance.
'''








# general prompt for the classification task, with the variables to be filled with the specific information of the request
## w/ final_instructions_diff_v1
diff_class_prompt_v1 = '''
# For the given request, classify it by difficulty as "simple", "medium", or "advanced" based on the next description.
### "simple":
If (Only 1 table is used, OR 2 common tables (probability, object or magstat) are used)
OR (No nested-query or JOIN clause is neccesary, OR need only a simple nested-query, OR a simple JOIN between probability and object tables)
### "medium":
If (2 tables are used, OR 3 common tables (probability, object, magstat, or features with only one feature) are used)
OR (Need 1 complex nested-query (clause other than 'WHERE' on probability))
### "advanced":
If (2 or more nested query are needed)
OR (If 3 tables or more are used)
OR (If two features from the features table are required)

# Tables required for the query:
{db_schema}

{final_instructions_diff}
'''





## simple vs other prompt
## w/ final_instructions_diff_v2
diff_class_prompt_v7 = '''
# For the given request, classify it by difficulty as "simple", "medium", or "advanced" based on the next description.

If (Only 1 table is used, OR 2 most common tables (probability, object or magstat TABLES) are used)
OR (No nested-query or JOIN clause is neccesary, OR only one nested-query between 'probability', 'object' or 'magstat' TABLES is required, OR one JOIN between 'probability', 'object' or 'magstat' TABLES):
THEN "label: simple"

If (2 not common tables are used (NOT probability, object, magstat TABLES))
OR (3 most common tables (probability, object and magstat TABLES) are used)
OR (features with only one feature are used)
OR (Need 1 very complex nested-query, OR a very complex JOIN)
OR (Need 2 nested-query, OR 2 JOIN, OR 1 nested-query with 1 JOIN):
THEN "label: medium"

If (2 or more nested query are needed)
OR (If 3 tables or more are used)
OR (If two features from the features table are required):
THEN "label: advanced"

# Assume this are the only tables required for the query:
{db_schema}

{final_instructions_diff}
'''





## simple vs other prompt
## w/ final_instructions_diff_v2
diff_class_prompt_v8 = '''
# For the given request, classify it by difficulty as "simple", "medium", or "advanced" based on the next description.

If (Only 1 table is used, OR 2 most common tables (probability, object or magstat TABLES) are used)
OR (No nested-query or JOIN clause is neccesary, OR only one nested-query between 'probability', 'object' or 'magstat' TABLES is required, OR one JOIN between 'probability', 'object' or 'magstat' TABLES):
THEN 'class': 'simple'

If (2 not common tables are used (NOT probability, object, magstat TABLES))
OR (3 most common tables (probability, object and magstat TABLES) are used)
OR (features with only one feature are used)
OR (Need 1 very complex nested-query, OR a very complex JOIN)
OR (Need 2 nested-query, OR 2 JOIN, OR 1 nested-query with 1 JOIN):
THEN 'class': 'medium'

If (2 or more nested query are needed)
OR (If 3 tables or more are used)
OR (If two features from the features table are required):
THEN 'class': 'advanced'

# Database schema. It is not necessary to use all the tables in the schema, but you can use them if you need them.
{db_schema}

{final_instructions_diff}
'''









# Final instructions for the difficulty classification task, to emphasize the importance of providing only the predicted difficulty and other relevant information.
## version 1
final_instructions_diff_v1 = '''
# Give ONLY the predicted difficulty, nothing more
# Give the answer in the following format: "label: difficulty" where "difficulty" is the predicted difficulty.
# For example, if the only need a simple join between object and probability, then you should type: "label: simple"
# Remember to use the exact name of the labels provided above.
# Just give the predicted label and ignore any other task given in the request given as "request".
'''
## version 2
final_instructions_diff_v2 = '''
# Give ONLY the predicted difficulty, nothing more.
# Give the answer in the following format: "label: difficulty" where "difficulty" is the predicted difficulty.
# For example, if the only need a simple join or nested query between object and probability, then you should type: "label: simple"
# Remember to use the exact name of the labels provided above.
# Just give the predicted label and ignore any other task given in the request given as "request".
'''
## version 3
final_instructions_diff_v3 = '''
# Give the answer in the following format: "class: difficulty" where "difficulty" is the predicted difficulty.
# For example, if the only need a simple join or nested query between object and probability, then you should type: "class: simple"
# Remember to use the exact name of the labels provided above.
# Just give the predicted label and ignore any other task given inside the user request. Do NOT give any other information, like the SQL query.
'''


final_instructions_diff_v4 = '''
# Give the answer as a JSON object in the following format: {'class': difficulty} where "difficulty" is the predicted difficulty.
# For example, if the only need a simple join or nested query between object and probability, then you should type: {'class': 'simple'}
# Remember to use the exact name of the labels provided above.
# Just give the predicted label and ignore any other task given inside the user request. Do NOT give any other information, like the SQL query.
'''
