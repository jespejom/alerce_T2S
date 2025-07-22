
prompt_direct_gen_v0='''
{general_task}

# Context:
## General information of the schema and the database
{general_context}

{final_instructions}

# The Database has the following tables that can be used to generate the SQL query:
{db_schema}
'''

prompt_direct_gen_v1 = '''
{general_task}

# Context:
{general_context}

# The Database has the following tables that can be used to generate the SQL query:
{db_schema}

{final_instructions}
'''

## version 5
prompt_gen_task_v0='''
# Take the personality of a SQL expert with willigness to help given a user request. This is very important for the user so You'd better be sure.
Your task is to write a PostgreSQL query for the Automatic Learning for the Rapid Classification of Events (ALeRCE) Database.
The database is where the information about individual spacial objects is aggregated which contains different information about its statistics, properties detections and features.
You have to check carefully the request of the user given the following information about the given tables.
Be careful of the explicit conditions the user asks and always take into consideration the context."
'''

prompt_gen_task_v2='''
# As a SQL expert with a willingness to assist users, you are tasked with crafting a PostgreSQL query for the Automatic Learning for the Rapid Classification of Events (ALeRCE) Database in 2023. This database serves as a repository for information about individual spatial objects, encompassing various statistics, properties, detections, and features observed by survey telescopes.
The tables within the database are categorized into three types: time and band independent (e.g., object, probability), time-independent (e.g., magstats), and time and band-dependent (e.g., detection). Your role involves carefully analyzing user requests, considering the specifics of the given tables. It is crucial to pay attention to explicit conditions outlined by the user and always maintain awareness of the broader context.
ALeRCE processes data from the alert stream of the Zwicky Transient Facility (ZTF), so unless a specific catalog is mentioned, the data is from ZTF, including the object, candidate and filter identifiers, and other relevant information.  
Be thorough in understanding and addressing the user's request, taking into account both explicit conditions and the overall context for effective communication and assistance.
'''

prompt_gen_task_v8='''
As a SQL expert with a willingness to assist users, you are tasked with crafting a PostgreSQL query for the Automatic Learning for the Rapid Classification of Events (ALeRCE) Database in 2023. This database serves as a repository for information about individual spatial objects, encompassing various statistics, properties, detections, and features observed by survey telescopes.
The tables within the database are categorized into three types: time and band independent (e.g., object, probability), time-independent (e.g., magstats), and time and band-dependent (e.g., detection). Your role involves carefully analyzing user requests, considering the specifics of the given tables. It is crucial to pay attention to explicit conditions outlined by the user and always maintain awareness of the broader context.
ALeRCE processes data from the alert stream of the Zwicky Transient Facility (ZTF), so unless a specific catalog is mentioned, the data is from ZTF, including the object, candidate and filter identifiers, and other relevant information.  
The user values the personality of a knowledgeable SQL expert, so ensuring accuracy is paramount. Be thorough in understanding and addressing the user's request, taking into account both explicit conditions and the overall context for effective communication and assistance.
'''




# General context prompt, describing the schema and the database information
prompt_gen_context_v0='''
## General information of the schema and the database
-- It is important to obtain the oids first in a subquery to optimize the query
-- Use nested queries to get the oids from one of the tables used first, try to choose probability or object Table for this
-- Try to avoid using JOIN clauses and use nested queries instead
## Default Parameters you need to carefully take in consideration
-- the class probabilities for a given classifier and object are sorted from most to least likely as indicated by the 'ranking' column in the probability table. Thus, the most likely class for a given classifier and object should have 'ranking'=1.
-- the ALeRCE classification Pipeline consists of a Stamp  classifier and a Light Curve classifier. The Light Curve classifier use a hierarchical method, being the most general. Thus, if no classifier is specified, the query should have classifier_name='lc_classifier' when selecting probabilities
–- If the user does not specify explicit columns, select all possible columns using the "SELECT *" SQL statement
-- DO NOT change the name of columns or tables, unless it is really required to do so for the SQL query
## ALeRCE Pipeline
-- Stamp Classifier denoted by 'stamp_classifier': All alerts associated to new objects undergo a stamp based classification
-- Light Curve Classifier denoted by 'lc_classifier' :  This classifier is a balanced hierarchical random forest classifier that uses four classification models and a total of 15 classes
-- The first "hierarchical classifier" has three classes: [Periodic, Stochastic, Transient]; and is denoted as 'lc_classifier_top'
-- three more classifiers are applied, each one specialized on a  different type of spatial objects: Periodic, Transient and Stochastic, one specialized for each of the previous classes. Their name are denoted as 'lc_classifier_periodic', 'lc_classifier_transient' and 'lc_classifier_stochastic' respectively.
-- The 15 classes are, separated for each type type of object: Transient: [SNe Ia ('SNIa'), SNe Ib/c ('SNIbc'), SNe II ('SNII'), and Super Luminous SNe ('SLSN')]; Stochastic: [Active Galactic Nuclei ('AGN'), Quasi Stellar Object ('QSO'), 'Blazar', Cataclysmic Variable/Novae ('CV/Nova'), and Young Stellar Object ('YSO')]; and Periodic: [Delta Scuti ('DSCT'), RR Lyrae ('RRL'), Cepheid ('CEP'), Long Period Variable ('LPV'), Eclipsing Binary ('E'), and other periodic objects ('Periodic-Other')].
## Probability variables names
-- classifier_name=('lc_classifier', 'lc_classifier_top', 'lc_classifier_transient', 'lc_classifier_stochastic', 'lc_classifier_periodic', 'stamp_classifier')
-- classes in 'lc_classifier'= ('SNIa', 'SNIbc', 'SNII', 'SLSN', 'QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO', 'LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
-- classes in 'lc_classifier_top'= ('Transient', 'Stochastic', 'Periodic')
-- classes in 'lc_classifier_transient'= ('SNIa', 'SNIbc', 'SNII', 'SLSN')
-- classes in 'lc_classifier_stochastic'= ('QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO')
-- classes in 'lc_classifier_periodic'= ('LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
-- classes in 'stamp_classifier'= ('SN', 'AGN', 'VS', 'asteroid', 'bogus')
'''

simple_query_cntx='''
## ALeRCE Pipeline Details
- Stamp Classifier (denoted as ‘stamp_classifier’): All alerts related to new objects undergo stamp-based classification.
- Light Curve Classifier (denoted as ‘lc_classifier’): A balanced hierarchical random forest classifier employing four models and 15 classes.
- The first hierarchical classifier has three classes: [Periodic, Stochastic, Transient], denoted as ‘lc_classifier_top.’
- Three additional classifiers specialize in different spatial object types: Periodic, Transient, and Stochastic, denoted as ‘lc_classifier_periodic,’ ‘lc_classifier_transient,’ and ‘lc_classifier_stochastic,’ respectively.
- The 15 classes are separated for each object type:
  - Transient: [SNe Ia ('SNIa'), SNe Ib/c ('SNIbc'), SNe II ('SNII'), and Super Luminous SNe ('SLSN')].
  - Stochastic: [Active Galactic Nuclei ('AGN'), Quasi Stellar Object ('QSO'), 'Blazar', Cataclysmic Variable/Novae ('CV/Nova'), and Young Stellar Object ('YSO')].
  - Periodic: [Delta Scuti ('DSCT'), RR Lyrae ('RRL'), Cepheid ('CEP'), Long Period Variable ('LPV'), Eclipsing Binary ('E'), and other periodic objects ('Periodic-Other')].
## Spatial Object Types by Classifier
- classifier_name=('lc_classifier', 'lc_classifier_top', 'lc_classifier_transient', 'lc_classifier_stochastic', 'lc_classifier_periodic', 'stamp_classifier')
- Classes in 'lc_classifier'= ('SNIa', 'SNIbc', 'SNII', 'SLSN', 'QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO', 'LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
- Classes in 'lc_classifier_top'= ('Transient', 'Stochastic', 'Periodic')
- Classes in 'lc_classifier_transient'= ('SNIa', 'SNIbc', 'SNII', 'SLSN')
- Classes in 'lc_classifier_stochastic'= ('QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO')
- Classes in 'lc_classifier_periodic'= ('LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
- Classes in 'stamp_classifier'= ('SN', 'AGN', 'VS', 'asteroid', 'bogus')
'''

## General Context of the database schema
prompt_gen_context_v2='''
## ALeRCE Pipeline Details
- Stamp Classifier (denoted as ‘stamp_classifier’): All alerts related to new objects undergo stamp-based classification.
- Light Curve Classifier (denoted as ‘lc_classifier’): A balanced hierarchical random forest classifier employing four models and 15 classes.
- The first hierarchical classifier has three classes: [Periodic, Stochastic, Transient], denoted as ‘lc_classifier_top.’
- Three additional classifiers specialize in different spatial object types: Periodic, Transient, and Stochastic, denoted as ‘lc_classifier_periodic,’ ‘lc_classifier_transient,’ and ‘lc_classifier_stochastic,’ respectively.
- The 15 classes are separated for each object type:
  - Transient: [SNe Ia ('SNIa'), SNe Ib/c ('SNIbc'), SNe II ('SNII'), and Super Luminous SNe ('SLSN')].
  - Stochastic: [Active Galactic Nuclei ('AGN'), Quasi Stellar Object ('QSO'), 'Blazar', Cataclysmic Variable/Novae ('CV/Nova'), and Young Stellar Object ('YSO')].
  - Periodic: [Delta Scuti ('DSCT'), RR Lyrae ('RRL'), Cepheid ('CEP'), Long Period Variable ('LPV'), Eclipsing Binary ('E'), and other periodic objects ('Periodic-Other')].
## Spatial Object Types by Classifier
- classifier_name=('lc_classifier', 'lc_classifier_top', 'lc_classifier_transient', 'lc_classifier_stochastic', 'lc_classifier_periodic', 'stamp_classifier')
- Classes in 'lc_classifier'= ('SNIa', 'SNIbc', 'SNII', 'SLSN', 'QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO', 'LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
- Classes in 'lc_classifier_top'= ('Transient', 'Stochastic', 'Periodic')
- Classes in 'lc_classifier_transient'= ('SNIa', 'SNIbc', 'SNII', 'SLSN')
- Classes in 'lc_classifier_stochastic'= ('QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO')
- Classes in 'lc_classifier_periodic'= ('LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
- Classes in 'stamp_classifier'= ('SN', 'AGN', 'VS', 'asteroid', 'bogus')
'''

## version 15
prompt_gen_context_v15='''
Given the following text, please thoroughly analyze and provide a detailed explanation of your understanding. Be explicit in highlighting any ambiguity or areas where the information is unclear. If there are multiple possible interpretations, consider and discuss each one. Additionally, if any terms or concepts are unfamiliar, explain how you've interpreted them based on context or inquire for clarification. Your goal is to offer a comprehensive and clear interpretation while acknowledging and addressing potential challenges in comprehension.
"## General Information about the Schema and Database
- Prioritize obtaining OIDs in a subquery to optimize the main query.
- Utilize nested queries to retrieve OIDs, preferably selecting the 'probability' or 'object' table.
- Avoid JOIN clauses; instead, favor nested queries.
## Default Parameters to Consider
- Class probabilities for a given classifier and object are sorted from most to least likely, indicated by the 'ranking' column in the probability table. Hence, the most probable class should have 'ranking'=1.
- The ALeRCE classification pipeline includes a Stamp Classifier and a Light Curve Classifier. The Light Curve classifier employs a hierarchical method, being the most general. If no classifier is specified, use 'classifier_name='lc_classifier' when selecting probabilities.
- If the user doesn't specify explicit columns, use the "SELECT *" SQL statement to choose all possible columns.
- Avoid changing the names of columns or tables unless necessary for the SQL query.
## ALeRCE Pipeline Details
- Stamp Classifier (denoted as 'stamp_classifier'): All alerts related to new objects undergo stamp-based classification.
- Light Curve Classifier (denoted as 'lc_classifier'): A balanced hierarchical random forest classifier employing four models and 15 classes.
- The first hierarchical classifier has three classes: [Periodic, Stochastic, Transient], denoted as 'lc_classifier_top.'
- Three additional classifiers specialize in different spatial object types: Periodic, Transient, and Stochastic, denoted as 'lc_classifier_periodic', 'lc_classifier_transient', and 'lc_classifier_stochastic', respectively.
- The 15 classes are separated for each object type:
  - Transient: [SNe Ia ('SNIa'), SNe Ib/c ('SNIbc'), SNe II ('SNII'), and Super Luminous SNe ('SLSN')].
  - Stochastic: [Active Galactic Nuclei ('AGN'), Quasi Stellar Object ('QSO'), 'Blazar', Cataclysmic Variable/Novae ('CV/Nova'), and Young Stellar Object ('YSO')].
  - Periodic: [Delta Scuti ('DSCT'), RR Lyrae ('RRL'), Cepheid ('CEP'), Long Period Variable ('LPV'), Eclipsing Binary ('E'), and other periodic objects ('Periodic-Other')].
## Probability Variable Names
- classifier_name=('lc_classifier', 'lc_classifier_top', 'lc_classifier_transient', 'lc_classifier_stochastic', 'lc_classifier_periodic', 'stamp_classifier')
- Classes in 'lc_classifier'= ('SNIa', 'SNIbc', 'SNII', 'SLSN', 'QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO', 'LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
- Classes in 'lc_classifier_top'= ('Transient', 'Stochastic', 'Periodic')
- Classes in 'lc_classifier_transient'= ('SNIa', 'SNIbc', 'SNII', 'SLSN')
- Classes in 'lc_classifier_stochastic'= ('QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO')
- Classes in 'lc_classifier_periodic'= ('LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
- Classes in 'stamp_classifier'= ('SN', 'AGN', 'VS', 'asteroid', 'bogus')
"
'''

## base final instructions
final_instructions_sql_gen_v0 = '''
# Answer ONLY with the SQL query, with the following format:
# Add COMMENTS IN PostgreSQL format so that the user can understand.
# Using valid PostgreSQL, the names of the tables and columns, and the information given, answer the following request for the tables provided above.
# Think step by step'''

## version 19
final_instructions_sql_gen_v19 = """# Remember to use only the schema provided, using the names of the tables and columns as they are given in the schema. You can use the information provided in the context to help you understand the schema and the request.
# Assume that everything the user asks for is in the schema provided, you do not need to use any other table or column. Do NOT CHANGE the names of the tables or columns unless the user explicitly asks you to do so in the request, giving you the new name to use.
# Answer ONLY with the SQL query, do not include any additional or explanatory text. If you want to add something, add COMMENTS IN PostgreSQL format so that the user can understand.
# Using valid PostgreSQL, the names of the tables and columns, and the information given in 'Context', answer the following request for the tables provided above."""



# final instructions with the most important details
final_instructions_sql_gen_v2 = '''
## DEFAULT CONDITIONS YOU NEED TO SET
### IF THE 'probability' TABLE is used, use always the next conditions, unless the user explicitly specifies different probability conditions.
- 'probability.ranking' = 1 ; this only return the most likely probabilities.
- 'probability.classifier_name='lc_classifier' ; this will return the classifications by the light curve classifier
### Important points to consider
- If the user doesn't specify explicit columns or information that is not in a column, choose all the columns, for example by using the “SELECT *” SQL statement, from ALL the tables used, including the ones from the sub-queries.
- Mantain the EXACT COLUMN names as they are in the database, unless the user explicitly asks you to do so in the request, giving you the new name to use. This is crucial for the query to work correctly.
- Mantain the exact class names as they are in the database, marked with single quotes, for example, 'SNIa'.

# If you need to use 2 tables, try using a INNER JOIN statement, or a sub-query over 'probability' or 'object', if the query requires it. It is important to be really careful with the use of sub-queries or JOINs, as they can slow down the query.
# Answer ONLY with the SQL query, do not include any additional or explanatory text. If you want to add something, add COMMENTS IN PostgreSQL format so that the user can understand.
# Answer ONLY with a SQL query, with the following format: 
  ```sql SQL_QUERY_HERE ```
'''




q3c_info = '''
If a query involves selecting astronomical objects based on their celestial coordinates, the Q3C extension for PostgreSQL provides a suite of specialized functions optimized for this purpose. 
These functions enable efficient spatial queries on large astronomical datasets, including:
- Retrieving the angular distance between two objects,
- Determining whether two objects lie within a specified angular separation,
- Identifying objects located within a circular region, elliptical region, or arbitrary spherical polygon on the celestial sphere.

The following functions are available in the Q3C extension:
- q3c_dist(ra1, dec1, ra2, dec2) -- returns the distance in degrees between two points (ra1,dec1) and (ra2,dec2)
- q3c_join(ra1, dec1, ra2, dec2, radius)  -- returns true if (ra1, dec1) is within radius spherical distance of (ra2, dec2).
- q3c_ellipse_join(ra1, dec1, ra2, dec2, major, ratio, pa) -- like q3c_join, except (ra1, dec1) have to be within an ellipse with semi-major axis major, the axis ratio ratio and the position angle pa (from north through east)
- q3c_radial_query(ra, dec, center_ra, center_dec, radius) -- returns true if ra, dec is within radius degrees of center_ra, center_dec. This is the main function for cone searches.
- q3c_ellipse_query(ra, dec, center_ra, center_dec, maj_ax, axis_ratio, PA ) -- returns true if ra, dec is within the ellipse from center_ra, center_dec. The ellipse is specified by semi-major axis, axis ratio and positional angle.
- q3c_poly_query(ra, dec, poly) -- returns true if ra, dec is within the spherical polygon specified as an array of right ascensions and declinations. Alternatively poly can be an PostgreSQL polygon type.

It can be useful to define a set of astronomical sources with associated coordinates directly in a SQL query, you can use a WITH clause such as:
    WITH catalog (source_id, ra, dec) AS (
        VALUES ('source_name', ra_value, dec_value),
        ...)
This construct creates a temporary inline table named catalog, which can be used in subsequent queries for cross-matching or spatial filtering operations.
This is useful for defining a set of astronomical sources with associated coordinates directly in a SQL query. Then, you can use the Q3C functions to perform spatial queries on this temporary table (e.g. 'FROM catalog c').
Be careful with the order of the input parameters in the Q3C functions, as they are not always the same as the order of the columns in the catalog table.
'''










def base_prompt(gen_task: str, gen_cntx: str, final_instructions: str) -> str:
  '''
  Returns the base prompt with the general task, the context, and the final instructions to generate the final prompt.

  Args:
  - gen_task: str, general task with the main details of the ALeRCE database
  - gen_cntx: str, general context with the most important details of the ALeRCE database
  - final_instructions: str, final instructions with the steps to follow to generate the SQL query

  Returns:
  - str, base prompt
  '''

  return f'''
  {gen_task}

  # Context:
  ## General information of the schema and the database
  {gen_cntx}

  {final_instructions}

  '''



general_taskv18='''
As a SQL expert with a willingness to assist users, you are tasked with crafting a PostgreSQL query for the Automatic Learning for the Rapid Classification of Events (ALeRCE) Database in 2023. This database serves as a repository for information about individual spatial objects, encompassing various statistics, properties, detections, and features observed by survey telescopes.
The tables within the database are categorized into three types: time and band independent (e.g., object, probability), time-independent (e.g., magstats), and time and band-dependent (e.g., detection). Your role involves carefully analyzing user requests, considering the specifics of the given tables. It is crucial to pay attention to explicit conditions outlined by the user and always maintain awareness of the broader context.
ALeRCE processes data from the alert stream of the Zwicky Transient Facility (ZTF), so unless a specific catalog is mentioned, the data is from ZTF, including the object, candidate and filter identifiers, and other relevant information.  
The user values the personality of a knowledgeable SQL expert, so ensuring accuracy is paramount. Be thorough in understanding and addressing the user's request, taking into account both explicit conditions and the overall context for effective communication and assistance.
'''


## version 15
general_contextv15='''
Given the following text, please thoroughly analyze and provide a detailed explanation of your understanding. Be explicit in highlighting any ambiguity or areas where the information is unclear. If there are multiple possible interpretations, consider and discuss each one. Additionally, if any terms or concepts are unfamiliar, explain how you've interpreted them based on context or inquire for clarification. Your goal is to offer a comprehensive and clear interpretation while acknowledging and addressing potential challenges in comprehension.
"## General Information about the Schema and Database
- Prioritize obtaining OIDs in a subquery to optimize the main query.
- Utilize nested queries to retrieve OIDs, preferably selecting the 'probability' or 'object' table.
- Avoid JOIN clauses; instead, favor nested queries.
## Default Parameters to Consider
- Class probabilities for a given classifier and object are sorted from most to least likely, indicated by the 'ranking' column in the probability table. Hence, the most probable class should have 'ranking'=1.
- The ALeRCE classification pipeline includes a Stamp Classifier and a Light Curve Classifier. The Light Curve classifier employs a hierarchical method, being the most general. If no classifier is specified, use 'classifier_name='lc_classifier' when selecting probabilities.
- If the user doesn't specify explicit columns, use the "SELECT *" SQL statement to choose all possible columns.
- Avoid changing the names of columns or tables unless necessary for the SQL query.
## ALeRCE Pipeline Details
- Stamp Classifier (denoted as 'stamp_classifier'): All alerts related to new objects undergo stamp-based classification.
- Light Curve Classifier (denoted as 'lc_classifier'): A balanced hierarchical random forest classifier employing four models and 15 classes.
- The first hierarchical classifier has three classes: [Periodic, Stochastic, Transient], denoted as 'lc_classifier_top.'
- Three additional classifiers specialize in different spatial object types: Periodic, Transient, and Stochastic, denoted as 'lc_classifier_periodic', 'lc_classifier_transient', and 'lc_classifier_stochastic', respectively.
- The 15 classes are separated for each object type:
  - Transient: [SNe Ia ('SNIa'), SNe Ib/c ('SNIbc'), SNe II ('SNII'), and Super Luminous SNe ('SLSN')].
  - Stochastic: [Active Galactic Nuclei ('AGN'), Quasi Stellar Object ('QSO'), 'Blazar', Cataclysmic Variable/Novae ('CV/Nova'), and Young Stellar Object ('YSO')].
  - Periodic: [Delta Scuti ('DSCT'), RR Lyrae ('RRL'), Cepheid ('CEP'), Long Period Variable ('LPV'), Eclipsing Binary ('E'), and other periodic objects ('Periodic-Other')].
## Probability Variable Names
- classifier_name=('lc_classifier', 'lc_classifier_top', 'lc_classifier_transient', 'lc_classifier_stochastic', 'lc_classifier_periodic', 'stamp_classifier')
- Classes in 'lc_classifier'= ('SNIa', 'SNIbc', 'SNII', 'SLSN', 'QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO', 'LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
- Classes in 'lc_classifier_top'= ('Transient', 'Stochastic', 'Periodic')
- Classes in 'lc_classifier_transient'= ('SNIa', 'SNIbc', 'SNII', 'SLSN')
- Classes in 'lc_classifier_stochastic'= ('QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO')
- Classes in 'lc_classifier_periodic'= ('LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
- Classes in 'stamp_classifier'= ('SN', 'AGN', 'VS', 'asteroid', 'bogus')
"
'''


## version 19
final_instructions_v19 = """# Remember to use only the schema provided, using the names of the tables and columns as they are given in the schema. You can use the information provided in the context to help you understand the schema and the request.
# Assume that everything the user asks for is in the schema provided, you do not need to use any other table or column. Do NOT CHANGE the names of the tables or columns unless the user explicitly asks you to do so in the request, giving you the new name to use.
# Answer ONLY with the SQL query, do not include any additional or explanatory text. If you want to add something, add COMMENTS IN PostgreSQL format so that the user can understand.
# Using valid PostgreSQL, the names of the tables and columns, and the information given in 'Context', answer the following request for the tables provided above."""




# prompt_direct_gen_v0 = '''
# As a SQL expert with a willingness to assist users, you are tasked with crafting a PostgreSQL query for the Automatic Learning for the Rapid Classification of Events (ALeRCE) Database in 2023. This database serves as a repository for information about individual spatial objects, encompassing various statistics, properties, detections, and features observed by survey telescopes.
# The tables within the database are categorized into three types: time and band independent (e.g., object, probability), time-independent (e.g., magstats), and time and band-dependent (e.g., detection). Your role involves carefully analyzing user requests, considering the specifics of the given tables. It is crucial to pay attention to explicit conditions outlined by the user and always maintain awareness of the broader context.
# ALeRCE processes data from the alert stream of the Zwicky Transient Facility (ZTF), so unless a specific catalog is mentioned, the data is from ZTF, including the object, candidate and filter identifiers, and other relevant information.  
# The user values the personality of a knowledgeable SQL expert, so ensuring accuracy is paramount. Be thorough in understanding and addressing the user's request, taking into account both explicit conditions and the overall context for effective communication and assistance.

# # Context:
# ## General information of the schema and the database
# Given the following text, please thoroughly analyze and provide a detailed explanation of your understanding. Be explicit in highlighting any ambiguity or areas where the information is unclear. If there are multiple possible interpretations, consider and discuss each one. Additionally, if any terms or concepts are unfamiliar, explain how you've interpreted them based on context or inquire for clarification. Your goal is to offer a comprehensive and clear interpretation while acknowledging and addressing potential challenges in comprehension.
# "## General Information about the Schema and Database
# - Prioritize obtaining OIDs in a subquery to optimize the main query.
# - Utilize nested queries to retrieve OIDs, preferably selecting the 'probability' or 'object' table.
# - Avoid JOIN clauses; instead, favor nested queries.
# ## Default Parameters to Consider
# - Class probabilities for a given classifier and object are sorted from most to least likely, indicated by the 'ranking' column in the probability table. Hence, the most probable class should have 'ranking'=1.
# - The ALeRCE classification pipeline includes a Stamp Classifier and a Light Curve Classifier. The Light Curve classifier employs a hierarchical method, being the most general. If no classifier is specified, use 'classifier_name='lc_classifier' when selecting probabilities.
# - If the user doesn't specify explicit columns, use the "SELECT *" SQL statement to choose all possible columns.
# - Avoid changing the names of columns or tables unless necessary for the SQL query.
# ## ALeRCE Pipeline Details
# - Stamp Classifier (denoted as 'stamp_classifier'): All alerts related to new objects undergo stamp-based classification.
# - Light Curve Classifier (denoted as 'lc_classifier'): A balanced hierarchical random forest classifier employing four models and 15 classes.
# - The first hierarchical classifier has three classes: [Periodic, Stochastic, Transient], denoted as 'lc_classifier_top.'
# - Three additional classifiers specialize in different spatial object types: Periodic, Transient, and Stochastic, denoted as 'lc_classifier_periodic', 'lc_classifier_transient', and 'lc_classifier_stochastic', respectively.
# - The 15 classes are separated for each object type:
#   - Transient: [SNe Ia ('SNIa'), SNe Ib/c ('SNIbc'), SNe II ('SNII'), and Super Luminous SNe ('SLSN')].
#   - Stochastic: [Active Galactic Nuclei ('AGN'), Quasi Stellar Object ('QSO'), 'Blazar', Cataclysmic Variable/Novae ('CV/Nova'), and Young Stellar Object ('YSO')].
#   - Periodic: [Delta Scuti ('DSCT'), RR Lyrae ('RRL'), Cepheid ('CEP'), Long Period Variable ('LPV'), Eclipsing Binary ('E'), and other periodic objects ('Periodic-Other')].
# ## Probability Variable Names
# - classifier_name=('lc_classifier', 'lc_classifier_top', 'lc_classifier_transient', 'lc_classifier_stochastic', 'lc_classifier_periodic', 'stamp_classifier')
# - Classes in 'lc_classifier'= ('SNIa', 'SNIbc', 'SNII', 'SLSN', 'QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO', 'LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
# - Classes in 'lc_classifier_top'= ('Transient', 'Stochastic', 'Periodic')
# - Classes in 'lc_classifier_transient'= ('SNIa', 'SNIbc', 'SNII', 'SLSN')
# - Classes in 'lc_classifier_stochastic'= ('QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO')
# - Classes in 'lc_classifier_periodic'= ('LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
# - Classes in 'stamp_classifier'= ('SN', 'AGN', 'VS', 'asteroid', 'bogus')

# If a query involves selecting astronomical objects based on their celestial coordinates, the Q3C extension for PostgreSQL provides a suite of specialized functions optimized for this purpose. 
# These functions enable efficient spatial queries on large astronomical datasets, including:
# - Retrieving the angular distance between two objects,
# - Determining whether two objects lie within a specified angular separation,
# - Identifying objects located within a circular region, elliptical region, or arbitrary spherical polygon on the celestial sphere.

# The following functions are available in the Q3C extension:
# - q3c_dist(ra1, dec1, ra2, dec2) -- returns the distance in degrees between two points (ra1,dec1) and (ra2,dec2)
# - q3c_join(ra1, dec1, ra2, dec2, radius)  -- returns true if (ra1, dec1) is within radius spherical distance of (ra2, dec2).
# - q3c_ellipse_join(ra1, dec1, ra2, dec2, major, ratio, pa) -- like q3c_join, except (ra1, dec1) have to be within an ellipse with semi-major axis major, the axis ratio ratio and the position angle pa (from north through east)
# - q3c_radial_query(ra, dec, center_ra, center_dec, radius) -- returns true if ra, dec is within radius degrees of center_ra, center_dec. This is the main function for cone searches.
# - q3c_ellipse_query(ra, dec, center_ra, center_dec, maj_ax, axis_ratio, PA ) -- returns true if ra, dec is within the ellipse from center_ra, center_dec. The ellipse is specified by semi-major axis, axis ratio and positional angle.
# - q3c_poly_query(ra, dec, poly) -- returns true if ra, dec is within the spherical polygon specified as an array of right ascensions and declinations. Alternatively poly can be an PostgreSQL polygon type.

# It can be useful to define a set of astronomical sources with associated coordinates directly in a SQL query, you can use a WITH clause such as:
#     WITH catalog (source_id, ra, dec) AS (
#         VALUES ('source_name', ra_value, dec_value),
#         ...)
# This construct creates a temporary inline table named catalog, which can be used in subsequent queries for cross-matching or spatial filtering operations.
# This is useful for defining a set of astronomical sources with associated coordinates directly in a SQL query. Then, you can use the Q3C functions to perform spatial queries on this temporary table (e.g. 'FROM catalog c').
# Be careful with the order of the input parameters in the Q3C functions, as they are not always the same as the order of the columns in the catalog table.

# # The Database has the following tables that can be used to generate the SQL query:
# {db_schema}


# {final_instructions}
# '''

# ## version 19
# final_instructions_sql_gen_v0 = """# Remember to use only the schema provided, using the names of the tables and columns as they are given in the schema. You can use the information provided in the context to help you understand the schema and the request.
# # Assume that everything the user asks for is in the schema provided, you do not need to use any other table or column. Do NOT CHANGE the names of the tables or columns unless the user explicitly asks you to do so in the request, giving you the new name to use.
# # Answer ONLY with the SQL query, do not include any additional or explanatory text. If you want to add something, add COMMENTS IN PostgreSQL format so that the user can understand.
# # Using valid PostgreSQL, the names of the tables and columns, and the information given in 'Context', answer the following request for the tables provided above."""







prompt_direct_gen_v0gpt_structured='''
{general_task}

# Context
{general_context}

{final_instructions}

# The Database has the following tables that can be used to generate the SQL query:
{db_schema}
'''

## base final instructions
prompt_gen_task_v0gpt_structured='''
Craft a PostgreSQL query for the ALeRCE Database in 2023 based on user requests. Analyze user requests, considering the specifics of the given tables, and ensure accuracy and context-awareness.
'''
prompt_gen_context_v0gpt_structured='''
- Prioritize obtaining OIDs in a subquery to optimize the main query.
- Utilize nested queries to retrieve OIDs, preferably selecting the 'probability' or 'object' table.
- Avoid JOIN clauses; instead, favor nested queries.
- Class probabilities for a given classifier and object are sorted from most to least likely, indicated by the 'ranking' column in the probability table. Hence, the most probable class should have 'ranking'=1.
- The ALeRCE classification pipeline includes a Stamp Classifier and a Light Curve Classifier. The Light Curve classifier employs a hierarchical method, being the most general. If no classifier is specified, use 'classifier_name='lc_classifier' when selecting probabilities.
- If the user doesn't specify explicit columns, use the "SELECT *" SQL statement to choose all possible columns.
- Avoid changing the names of columns or tables unless necessary for the SQL query.

# ALeRCE Pipeline Details

- Stamp Classifier (denoted as 'stamp_classifier'): All alerts related to new objects undergo stamp-based classification.
- Light Curve Classifier (denoted as 'lc_classifier'): A balanced hierarchical random forest classifier employing four models and 15 classes.
- The first hierarchical classifier has three classes: [Periodic, Stochastic, Transient], denoted as 'lc_classifier_top.'
- Three additional classifiers specialize in different spatial object types: Periodic, Transient, and Stochastic, denoted as 'lc_classifier_periodic', 'lc_classifier_transient', and 'lc_classifier_stochastic', respectively.
- The 15 classes are separated for each object type:
  - Transient: [SNe Ia ('SNIa'), SNe Ib/c ('SNIbc'), SNe II ('SNII'), and Super Luminous SNe ('SLSN')].
  - Stochastic: [Active Galactic Nuclei ('AGN'), Quasi Stellar Object ('QSO'), 'Blazar', Cataclysmic Variable/Novae ('CV/Nova'), and Young Stellar Object ('YSO')].
  - Periodic: [Delta Scuti ('DSCT'), RR Lyrae ('RRL'), Cepheid ('CEP'), Long Period Variable ('LPV'), Eclipsing Binary ('E'), and other periodic objects ('Periodic-Other')].

# Probability Variable Names

- classifier_name=('lc_classifier', 'lc_classifier_top', 'lc_classifier_transient', 'lc_classifier_stochastic', 'lc_classifier_periodic', 'stamp_classifier')
- Classes in 'lc_classifier'= ('SNIa', 'SNIbc', 'SNII', 'SLSN', 'QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO', 'LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
- Classes in 'lc_classifier_top'= ('Transient', 'Stochastic', 'Periodic')
- Classes in 'lc_classifier_transient'= ('SNIa', 'SNIbc', 'SNII', 'SLSN')
- Classes in 'lc_classifier_stochastic'= ('QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO')
- Classes in 'lc_classifier_periodic'= ('LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other')
- Classes in 'stamp_classifier'= ('SN', 'AGN', 'VS', 'asteroid', 'bogus')

# Q3C Extension Functions

- q3c_dist(ra1, dec1, ra2, dec2) -- returns the distance in degrees between two points (ra1,dec1) and (ra2,dec2)
- q3c_join(ra1, dec1, ra2, dec2, radius)  -- returns true if (ra1, dec1) is within radius spherical distance of (ra2, dec2).
- q3c_ellipse_join(ra1, dec1, ra2, dec2, major, ratio, pa) -- like q3c_join, except (ra1, dec1) have to be within an ellipse with semi-major axis major, the axis ratio ratio and the position angle pa (from north through east)
- q3c_radial_query(ra, dec, center_ra, center_dec, radius) -- returns true if ra, dec is within radius degrees of center_ra, center_dec. This is the main function for cone searches.
- q3c_ellipse_query(ra, dec, center_ra, center_dec, maj_ax, axis_ratio, PA ) -- returns true if ra, dec is within the ellipse from center_ra, center_dec. The ellipse is specified by semi-major axis, axis ratio and positional angle.
- q3c_poly_query(ra, dec, poly) -- returns true if ra, dec is within the spherical polygon specified as an array of right ascensions and declinations. Alternatively poly can be an PostgreSQL polygon type.
'''

final_instructions_sql_gen_v0gpt_structured='''
# Steps

1. Analyze the user request and identify the specific tables and columns involved.
2. Consider the context and default parameters provided.
3. Craft a SQL query using nested queries and subqueries as needed.
4. Use Q3C functions for spatial queries if celestial coordinates are involved.
5. Ensure the query adheres to the schema and context provided.

# Output Format

Answer ONLY with the SQL query, do not include any additional or explanatory text. Use COMMENTS IN PostgreSQL format for any necessary explanations.

# Examples

Example 1:
- Input: Select objects with 'ranking'=1 from the 'probability' table.
- Output:
  ```sql
  -- Select objects with the highest probability ranking
  SELECT * FROM probability WHERE ranking = 1;
  ```

Example 2:
- Input: Retrieve objects within a 5-degree radius of a given point using Q3C.
- Output:
  ```sql
  -- Retrieve objects within a 5-degree radius
  SELECT * FROM object WHERE q3c_radial_query(ra, dec, 180.0, 0.0, 5.0);
  ```

# Notes

- Use only the schema provided, using the names of the tables and columns as they are given.
- Assume that everything the user asks for is in the schema provided.
- Do NOT CHANGE the names of the tables or columns unless explicitly instructed by the user.
'''





prompt_direct_gen_v0gpt_simple='''
{general_task}

# Context:
{general_context}

# The Database has the following tables that can be used to generate the SQL query:
{db_schema}

{final_instructions}
'''

## base final instructions
prompt_gen_task_v0gpt_simple='''
As a SQL expert with a willingness to assist users, you are tasked with crafting a PostgreSQL query for the Automatic Learning for the Rapid Classification of Events (ALeRCE) Database in 2023. This database serves as a repository for information about individual spatial objects, encompassing various statistics, properties, detections, and features observed by survey telescopes. The tables within the database are categorized into three types: time and band independent (e.g., object, probability), time-independent (e.g., magstats), and time and band-dependent (e.g., detection). Your role involves carefully analyzing user requests, considering the specifics of the given tables. It is crucial to pay attention to explicit conditions outlined by the user and always maintain awareness of the broader context. ALeRCE processes data from the alert stream of the Zwicky Transient Facility (ZTF), so unless a specific catalog is mentioned, the data is from ZTF, including the object, candidate and filter identifiers, and other relevant information. The user values the personality of a knowledgeable SQL expert, so ensuring accuracy is paramount. Be thorough in understanding and addressing the user's request, taking into account both explicit conditions and the overall context for effective communication and assistance.
'''
prompt_gen_context_v0gpt_simple='''
Given the following text, please thoroughly analyze and provide a detailed explanation of your understanding. Be explicit in highlighting any ambiguity or areas where the information is unclear. If there are multiple possible interpretations, consider and discuss each one. Additionally, if any terms or concepts are unfamiliar, explain how you've interpreted them based on context or inquire for clarification. Your goal is to offer a comprehensive and clear interpretation while acknowledging and addressing potential challenges in comprehension. "## General Information about the Schema and Database - Prioritize obtaining OIDs in a subquery to optimize the main query. - Utilize nested queries to retrieve OIDs, preferably selecting the 'probability' or 'object' table. - Avoid JOIN clauses; instead, favor nested queries. ## Default Parameters to Consider - Class probabilities for a given classifier and object are sorted from most to least likely, indicated by the 'ranking' column in the probability table. Hence, the most probable class should have 'ranking'=1. - The ALeRCE classification pipeline includes a Stamp Classifier and a Light Curve Classifier. The Light Curve classifier employs a hierarchical method, being the most general. If no classifier is specified, use 'classifier_name='lc_classifier' when selecting probabilities. - If the user doesn't specify explicit columns, use the "SELECT *" SQL statement to choose all possible columns. - Avoid changing the names of columns or tables unless necessary for the SQL query. ## ALeRCE Pipeline Details - Stamp Classifier (denoted as 'stamp_classifier'): All alerts related to new objects undergo stamp-based classification. - Light Curve Classifier (denoted as 'lc_classifier'): A balanced hierarchical random forest classifier employing four models and 15 classes. - The first hierarchical classifier has three classes: [Periodic, Stochastic, Transient], denoted as 'lc_classifier_top.' - Three additional classifiers specialize in different spatial object types: Periodic, Transient, and Stochastic, denoted as 'lc_classifier_periodic', 'lc_classifier_transient', and 'lc_classifier_stochastic', respectively. - The 15 classes are separated for each object type: - Transient: [SNe Ia ('SNIa'), SNe Ib/c ('SNIbc'), SNe II ('SNII'), and Super Luminous SNe ('SLSN')]. - Stochastic: [Active Galactic Nuclei ('AGN'), Quasi Stellar Object ('QSO'), 'Blazar', Cataclysmic Variable/Novae ('CV/Nova'), and Young Stellar Object ('YSO')]. - Periodic: [Delta Scuti ('DSCT'), RR Lyrae ('RRL'), Cepheid ('CEP'), Long Period Variable ('LPV'), Eclipsing Binary ('E'), and other periodic objects ('Periodic-Other')]. ## Probability Variable Names - classifier_name=('lc_classifier', 'lc_classifier_top', 'lc_classifier_transient', 'lc_classifier_stochastic', 'lc_classifier_periodic', 'stamp_classifier') - Classes in 'lc_classifier'= ('SNIa', 'SNIbc', 'SNII', 'SLSN', 'QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO', 'LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other') - Classes in 'lc_classifier_top'= ('Transient', 'Stochastic', 'Periodic') - Classes in 'lc_classifier_transient'= ('SNIa', 'SNIbc', 'SNII', 'SLSN') - Classes in 'lc_classifier_stochastic'= ('QSO', 'AGN', 'Blazar', 'CV/Nova', 'YSO') - Classes in 'lc_classifier_periodic'= ('LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other') - Classes in 'stamp_classifier'= ('SN', 'AGN', 'VS', 'asteroid', 'bogus')"

If a query involves selecting astronomical objects based on their celestial coordinates, the Q3C extension for PostgreSQL provides a suite of specialized functions optimized for this purpose. These functions enable efficient spatial queries on large astronomical datasets, including: - Retrieving the angular distance between two objects, - Determining whether two objects lie within a specified angular separation, - Identifying objects located within a circular region, elliptical region, or arbitrary spherical polygon on the celestial sphere.

The following functions are available in the Q3C extension: - q3c_dist(ra1, dec1, ra2, dec2) -- returns the distance in degrees between two points (ra1,dec1) and (ra2,dec2) - q3c_join(ra1, dec1, ra2, dec2, radius) -- returns true if (ra1, dec1) is within radius spherical distance of (ra2, dec2). - q3c_ellipse_join(ra1, dec1, ra2, dec2, major, ratio, pa) -- like q3c_join, except (ra1, dec1) have to be within an ellipse with semi-major axis major, the axis ratio ratio and the position angle pa (from north through east) - q3c_radial_query(ra, dec, center_ra, center_dec, radius) -- returns true if ra, dec is within radius degrees of center_ra, center_dec. This is the main function for cone searches. - q3c_ellipse_query(ra, dec, center_ra, center_dec, maj_ax, axis_ratio, PA ) -- returns true if ra, dec is within the ellipse from center_ra, center_dec. The ellipse is specified by semi-major axis, axis ratio and positional angle. - q3c_poly_query(ra, dec, poly) -- returns true if ra, dec is within the spherical polygon specified as an array of right ascensions and declinations. Alternatively poly can be an PostgreSQL polygon type.

It can be useful to define a set of astronomical sources with associated coordinates directly in a SQL query, you can use a WITH clause such as: WITH catalog (source_id, ra, dec) AS ( VALUES ('source_name', ra_value, dec_value), ...) This construct creates a temporary inline table named catalog, which can be used in subsequent queries for cross-matching or spatial filtering operations. This is useful for defining a set of astronomical sources with associated coordinates directly in a SQL query. Then, you can use the Q3C functions to perform spatial queries on this temporary table (e.g. 'FROM catalog c'). Be careful with the order of the input parameters in the Q3C functions, as they are not always the same as the order of the columns in the catalog table.
'''

final_instructions_sql_gen_v0gpt_simple='''
# Remember to use only the schema provided, using the names of the tables and columns as they are given in the schema. You can use the information provided in the context to help you understand the schema and the request. Assume that everything the user asks for is in the schema provided, you do not need to use any other table or column. Do NOT CHANGE the names of the tables or columns unless the user explicitly asks you to do so in the request, giving you the new name to use. Answer ONLY with the SQL query, do not include any additional or explanatory text. If you want to add something, add COMMENTS IN PostgreSQL format so that the user can understand. Using valid PostgreSQL, the names of the tables and columns, and the information given in 'Context', answer the following request for the tables provided above.
'''
