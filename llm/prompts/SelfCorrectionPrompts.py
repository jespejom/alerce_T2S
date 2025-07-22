
# General Self-Correction Prompt
## Self-correcting task prompt
general_task_selfcorr_v1='''
As a SQL expert with a willingness to assist users, you are tasked with crafting a PostgreSQL query for the Automatic Learning for the Rapid Classification of Events (ALeRCE) Database in 2023. This database serves as a repository for information about individual spatial objects, encompassing various statistics, properties, detections, and features observed by survey telescopes.
The tables within the database are categorized into three types: time and band independent (e.g., object, probability), time-independent (e.g., magstats), and time and band-dependent (e.g., detection). Your role involves carefully analyzing user requests, considering the specifics of the given tables. It is crucial to pay attention to explicit conditions outlined by the user and always maintain awareness of the broader context.
The user values the personality of a knowledgeable SQL expert, so ensuring accuracy is paramount. Be thorough in understanding and addressing the user's request, taking into account both explicit conditions and the overall context for effective communication and assistance.
'''







## General Context of the database schema
general_context_selfcorr_v1='''
## ALeRCE Pipeline Details
- Stamp Classifier (denoted as ‘stamp_classifier’): All alerts related to new objects undergo stamp-based classification.
- Light Curve Classifier (denoted as ‘lc_classifier’): A balanced hierarchical random forest classifier employing four models and 15 classes.
- The first hierarchical classifier has three classes: [periodic, stochastic, transient], denoted as ‘lc_classifier_top.’
- Three additional classifiers specialize in different spatial object types: Periodic, Transient, and Stochastic, denoted as ‘lc_classifier_periodic,’ ‘lc_classifier_transient,’ and ‘lc_classifier_stochastic,’ respectively.
- The 15 classes are separated for each object type:
  - Transient: [SNe Ia ('SNIa'), SNe Ib/c ('SNIbc'), SNe II ('SNII'), and Super Luminous SNe ('SLSN')].
  - Stochastic: [Active Galactic Nuclei ('AGN'), Quasi Stellar Object ('QSO'), 'Blazar', Cataclysmic Variable/Novae ('CV/Nova'), and Young Stellar Object ('YSO')].
  - Periodic: [Delta Scuti ('DSCT'), RR Lyrae ('RRL'), Cepheid ('Ceph'), Long Period Variable ('LPV'), Eclipsing Binary ('E'), and other periodic objects ('Periodic-Other')].
## Spatial Object Types by Classifier
- classifier_name=('lc_classifier', 'lc_classifier_top', 'lc_classifier_transient', 'lc_classifier_stochastic', 'lc_classifier_periodic', 'stamp_classifier')
- Classes in 'lc_classifier'= ('SNIa', 'SNIbc', 'SNII', 'SLSN', 'QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO', 'LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
- Classes in 'lc_classifier_top'= ('transient', 'stochastic', 'periodic')
- Classes in 'lc_classifier_transient'= ('SNIa', 'SNIbc', 'SNII', 'SLSN')
- Classes in 'lc_classifier_stochastic'= ('QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO')
- Classes in 'lc_classifier_periodic'= ('LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
- Classes in 'stamp_classifier'= ('SN', 'AGN', 'VS', 'asteroid', 'bogus')
## DEFAULT CONDITIONS YOU NEED TO SET
### IF THE 'probability' TABLE is used, use always the next conditions, unless the user explicitly specifies different probability conditions.
- 'probability.ranking' = 1 ; this only return the most likely probabilities.
- 'probability.classifier_name='lc_classifier' ; this will return the classifications by the light curve classifier
### GENERAL
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the “SELECT *” SQL statement.
- Use the exact class names as they are in the database, marked with single quotes, for example, 'SNIa'.
# If you need to use 2 or 3 tables, try using a sub-query over 'probability' or 'object' if it is necessary (priority in this order).
'''


# Final Instructions, emphasizing the importance to correct the query and the format to answer
## version 1
final_instr_selfcorr_v1 = '''# Using valid PostgreSQL, the CORRECT the query given the error, using the correct database schema or nested queries to optimize.
# Answer ONLY with the SQL query
# Add COMMENTS IN PostgreSQL format so that the user can understand.
# SQL:
'''

final_instr_selfcorr_v0 = '''
# Check the query and correct it modifying the SQL code where the error is found.
# Add COMMENTS IN PostgreSQL format so that the user can understand.
# Answer ONLY with the SQL query
# SQL:
'''

self_correction_timeout_prompt_v1='''{Self_correction_task}
# Context:
## General information of the schema and the database
{gen_cntx}

# Correct a SQL query given the next user request:
{request}
# This are the tables Schema. Assume that only the next tables are required for the query:
{tab_schema}

# The next query is not working due to a timeout Error, correct the query using the correct database schema or nested queries to optimize.
# SQL Query
{sql_query}
# Error returned when executing the query in the ALeRCE database
{sql_error}

# Follow the next advices to correct the query:
- Check if the query is using the correct tables and columns, and if the conditions are correct, given the user request.
- Check if the SQL code includes all the requested conditions.

# If there are no problems with the previous steps, follow the next advices to correct the query:
- Check if the SQL code includes the necessary conditions to optimize the query, and if the query is using the correct database schema or nested queries to optimize.
    - It is possible that the query is too complex and it is necessary to use nested queries to optimize the query.
    - If there is a JOIN or a sub-query between object and probability, check if the condition 'ranking=1' is set in the probability table, unless the request said otherwise.
- Check if are at least 3 conditions over the probability table, because if not, the query is too general. Add more conditions if necessary.
- If there are conditions involving dates or times, check if the dates are not too far away, or are in a reasonable range.

{final_instructions}
'''


# Separate prompts for Timeout, No Exist and Schema Error
## version 2
## Timeout Self-Correction Prompt
self_correction_timeout_prompt_v2='''{Self_correction_task}

# Correct a SQL query given the next user request:
{request}

# These are the table schemas. Assume that only the following tables are required for the query:
{tab_schema}

# The following query is not working due to a timeout error, correct the query using the correct database schema or nested queries to optimize.
# SQL Query
{sql_query}
# Error returned when executing the query in the ALeRCE database
{sql_error}

# Follow these guidelines to correct the query:
- Check if the SQL code includes the necessary conditions to optimize the query, and if the query is using the correct database schema or nested queries to optimize.
    - It is possible that the query is too complex and it is necessary to use nested queries to optimize the query.
    - If there is a JOIN or a sub-query between some table and probability, check if the condition 'ranking=1' is set in the probability table, unless the request said otherwise.
- If there are conditions involving dates or times, check if the dates are not too far away, or are in a reasonable range.
# If the probability table is used, use always the next conditions, unless the user explicitly specifies different probability conditions.
- Ensure there are at least 3 conditions on the probability table, because if not, the query is too general. Add more conditions if necessary.

{final_instructions}
'''




## Non-existent Self-Correction Prompt
self_correction_no_exist_prompt_v2='''{Self_correction_task}

# Correct a SQL query given the next user request:
{request}

# These are the table schemas. Assume that only the following tables are required for the query:
{tab_schema}

# The following query is not working due to a non-existent Error, correct the query using the correct database schema or nested queries to optimize.
# SQL Query
{sql_query}
# Error returned when executing the query in the ALeRCE database
{sql_error}

# The query is not working because the table or column does not exist in the database schema. Check the table or column name and correct the query.
# Follow the next advices to correct the query:
- Check if the query is using the correct tables and columns provided, and if the conditions are correct, given the user request.
- Check if the SQL code includes all the requested conditions.
- If the error is due to a table or column that does not exist, check the table or column name and correct the query using the correct database schema provided.
- If the error is due to a function that does not exist, try to modify the query using only the information given in the database schema.
- If the error is due to a relation that does not exist, check the relation name and correct the query using the correct database schema.
# All the information needed is in the database schema, use only the information provided in the database schema to correct the query. If it is not explicitly provided, go for the most common sense approach.

{final_instructions}
'''

## Schema Error Self-Correction Prompt
self_correction_schema_prompt_v2='''{Self_correction_task}

# Correct a SQL query given the next user request:
{request}

# These are the table schemas. Assume that only the following tables are required for the query:
{tab_schema}

# The follosing query is not working due to a syntax error, correct the query using the correct database schema.
# SQL Query
{sql_query}
# Error returned when executing the query in the ALeRCE database
{sql_error}

# Follow these guidelines to correct the query:
- Check if the query is using the correct database schema. This includes the correct names of the tables and the correct names of the columns. If not, correct the query.
- Check if the query has the correct syntax. If not, correct the query.
- If there is a "missing FROM-clause entry", check where the table or sub-query is used in the query and add the correct name of the table or sub-query.
- Use only the information provided in the database schema to correct the query. If it is not explicitly provided, apply the most common sense approach.

{final_instructions}
'''




self_correction_other_prompt_v0='''{Self_correction_task}
# Context:
## General information of the schema and the database
{gen_cntx}
# Correct a SQL query given the next user request:
{request}

# This are the tables Schema. Assume that only the next tables are required for the query:
{tab_schema}

# The next query is not working due to a syntax error, correct the query using the correct database schema.
# SQL Query
{sql_query}
# Error returned when executing the query in the ALeRCE database
{sql_error}

# Follow the next advices to correct the query:
- Check if the query is using the correct database schema. This includes the correct names of the tables and the correct names of the columns. If not, correct the query.
- Check if the query have the correct syntax. If not, correct the query.
- If there is a "missing FROM-clause entry", check where the table or sub-query is used in the query and add the correct name of the table or sub-query.

{final_instructions}
'''