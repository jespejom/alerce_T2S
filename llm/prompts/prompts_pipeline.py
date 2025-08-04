from prompts import DBSchemaPrompts, SchemaLinkingPrompts
from prompts import (
    DBSchemaPrompts, SchemaLinkingPrompts, DiffClassificationPrompts, Q3cClassificationPrompts, SelfCorrectionPrompts,
    DirectSQLGenPrompts
)
from prompts.SbySPrompts import (
    SbySPlanPrompts, SbySSQLGenPrompts, SbySCoTPrompts
)

def get_prompt_version(
        prompt_version: str
        ) -> dict:
    """
    Get the prompt version for the pipeline.
    Args:
        prompt_version (str): The version of the prompt to use.
    Returns:
        dict: The prompt version dictionary.
    """
    if prompt_version not in prompt_versions.keys():
        raise ValueError(f"Prompt version {prompt_version} not found in the prompt versions dictionary.")
    return prompt_versions[prompt_version]
    

schema_linking_prompt_versions = {
    "sl_v0": {
        "db_schema": DBSchemaPrompts.alerce_tables_desc,
        "schema_linking_format": SchemaLinkingPrompts.prompt_schema_linking_v0,
        "sl_final_instructions": SchemaLinkingPrompts.sl_final_instructions_v1
    },
    "sl_v1": {
        "db_schema": DBSchemaPrompts.alerce_tables_desc,
        "schema_linking_format": SchemaLinkingPrompts.prompt_schema_linking_v0,
        "sl_final_instructions": SchemaLinkingPrompts.sl_final_instructions_v1
    },
    "sl_v2": {
        "db_schema": DBSchemaPrompts.alerce_tables_desc + "\n" + DBSchemaPrompts.ref_keys,
        "schema_linking_format": SchemaLinkingPrompts.prompt_schema_linking_v0,
        "sl_final_instructions": SchemaLinkingPrompts.sl_final_instructions_v1
    },
    "sl_v3": {
        "db_schema": DBSchemaPrompts.alerce_tables_desc,
        "schema_linking_format": SchemaLinkingPrompts.prompt_schema_linking_v0,
        "sl_final_instructions": SchemaLinkingPrompts.sl_final_instructions_v2
    }
}

diff_classification_prompt_versions = {
    "diff_v0": {
        "db_schema": DBSchemaPrompts.schema_all_cntxV2,
        "diff_class_format": DiffClassificationPrompts.diff_class_prompt_v1,
        "diff_final_instructions": DiffClassificationPrompts.final_instructions_diff_v1
    },
    "diff_v7": {
        "db_schema": DBSchemaPrompts.schema_all_cntxV1,
        "diff_class_format": DiffClassificationPrompts.diff_class_prompt_v7,
        "diff_final_instructions": DiffClassificationPrompts.final_instructions_diff_v2
    },
    "diff_v8": {
        "db_schema": DBSchemaPrompts.schema_all_cntxV2_indx,
        "diff_class_format": DiffClassificationPrompts.diff_class_prompt_v8,
        "diff_final_instructions": DiffClassificationPrompts.final_instructions_diff_v4
        }
}

dir_prompt_versions = {
    "dir_v0": {
    "db_schema": DBSchemaPrompts.schema_all_cntxV2,
    "sql_gen_prompt_format": DirectSQLGenPrompts.prompt_direct_gen_v0,
    "sql_gen_task": DirectSQLGenPrompts.prompt_gen_task_v0,
    "sql_gen_context": DirectSQLGenPrompts.prompt_gen_context_v0 + DirectSQLGenPrompts.q3c_info,
    "sql_gen_final_instructions": DirectSQLGenPrompts.final_instructions_sql_gen_v0
    },
    "dir_v8": {
    "db_schema": DBSchemaPrompts.schema_all_cntxV2_indx,
    "sql_gen_prompt_format": DirectSQLGenPrompts.prompt_direct_gen_v1,
    "sql_gen_task": DirectSQLGenPrompts.prompt_gen_task_v8,
    "sql_gen_context": DirectSQLGenPrompts.prompt_gen_context_v15 + DirectSQLGenPrompts.q3c_info,
    "sql_gen_final_instructions": DirectSQLGenPrompts.final_instructions_sql_gen_v19
    },
    "dir_v9": {
        "db_schema": DBSchemaPrompts.schema_all_cntxV2_indx,
        "sql_gen_prompt_format": DirectSQLGenPrompts.prompt_direct_gen_v0gpt_structured,
        "sql_gen_task": DirectSQLGenPrompts.prompt_gen_task_v0gpt_structured,
        "sql_gen_context": DirectSQLGenPrompts.prompt_gen_context_v0gpt_structured,
        "sql_gen_final_instructions": DirectSQLGenPrompts.final_instructions_sql_gen_v0gpt_structured
    },
    "dir_v10": {
        "db_schema": DBSchemaPrompts.schema_all_cntxV2_indx,
        "sql_gen_prompt_format": DirectSQLGenPrompts.prompt_direct_gen_v0gpt_simple,
        "sql_gen_task": DirectSQLGenPrompts.prompt_gen_task_v0gpt_simple,
        "sql_gen_context": DirectSQLGenPrompts.prompt_gen_context_v0gpt_simple,
        "sql_gen_final_instructions": DirectSQLGenPrompts.final_instructions_sql_gen_v0gpt_simple
    }
}


sbs_prompt_versions = {
    "sbs_v0": {
        "db_schema": DBSchemaPrompts.schema_all_cntxV2,
        "sql_gen_prompt_format": DirectSQLGenPrompts.prompt_direct_gen_v0,
        "sql_gen_task": DirectSQLGenPrompts.prompt_gen_task_v0,
        "sql_gen_context": DirectSQLGenPrompts.prompt_gen_context_v0 + DirectSQLGenPrompts.q3c_info,
        "sql_gen_final_instructions": DirectSQLGenPrompts.final_instructions_sql_gen_v0,
        "medium_plan_prompt_format": SbySPlanPrompts.medium_decomp_prompt_v0,
        "medium_plan_task": SbySPlanPrompts.medium_decomp_task_v3,
        "medium_plan_context": DirectSQLGenPrompts.prompt_gen_context_v0 + DirectSQLGenPrompts.q3c_info,
        "medium_plan_instructions": SbySPlanPrompts.medium_query_instructions_1_v2,
        "adv_plan_prompt_format": SbySPlanPrompts.adv_decomp_prompt_v0,
        "adv_plan_task": SbySPlanPrompts.adv_decomp_task_v3,
        "adv_plan_context": DirectSQLGenPrompts.prompt_gen_context_v0 + DirectSQLGenPrompts.q3c_info,
        "adv_plan_instructions": SbySPlanPrompts.adv_query_instructions_1_v3,
        "medium_sql_gen_prompt_format": SbySSQLGenPrompts.medium_sql_gen_prompt_v0,
        "medium_sql_gen_task": DirectSQLGenPrompts.prompt_gen_task_v0,
        "medium_sql_gen_context": DirectSQLGenPrompts.prompt_gen_context_v0 + DirectSQLGenPrompts.q3c_info,
        "medium_sql_gen_instructions": DirectSQLGenPrompts.final_instructions_sql_gen_v0,
        "adv_sql_gen_prompt_format": SbySSQLGenPrompts.adv_sql_gen_prompt_v0,
        "adv_sql_gen_task": DirectSQLGenPrompts.prompt_gen_task_v0,
        "adv_sql_gen_context": DirectSQLGenPrompts.prompt_gen_context_v0 + DirectSQLGenPrompts.q3c_info,
        "adv_sql_gen_instructions": DirectSQLGenPrompts.final_instructions_sql_gen_v0
    },
    "sbs_v4": {
        "db_schema": DBSchemaPrompts.schema_all_cntxV2_indx,
        "sql_gen_prompt_format": DirectSQLGenPrompts.prompt_direct_gen_v1,
        "sql_gen_task": DirectSQLGenPrompts.prompt_gen_task_v2,
        "sql_gen_context": DirectSQLGenPrompts.prompt_gen_context_v2 + DirectSQLGenPrompts.q3c_info,
        "sql_gen_final_instructions": DirectSQLGenPrompts.final_instructions_sql_gen_v2,

        "medium_plan_prompt_format": SbySPlanPrompts.medium_decomp_prompt_v0,
        "medium_plan_task": SbySPlanPrompts.medium_decomp_task_v3 + SbySPlanPrompts.gpt4turbo1106_decomposed_prompt_2,
        "medium_plan_context": DirectSQLGenPrompts.prompt_gen_context_v2 + DirectSQLGenPrompts.q3c_info,
        "medium_plan_instructions": SbySPlanPrompts.medium_query_instructions_1_v2,

        "adv_plan_prompt_format": SbySPlanPrompts.adv_decomp_prompt_v0,
        "adv_plan_task": SbySPlanPrompts.adv_decomp_task_v3 + SbySPlanPrompts.gpt4turbo1106_decomposed_prompt_2,
        "adv_plan_context": DirectSQLGenPrompts.prompt_gen_context_v2 + DirectSQLGenPrompts.q3c_info,
        "adv_plan_instructions": SbySPlanPrompts.adv_query_instructions_1_v3,

        "medium_sql_gen_prompt_format": SbySSQLGenPrompts.medium_sql_gen_prompt_v0,
        "medium_sql_gen_task": DirectSQLGenPrompts.prompt_gen_task_v2,
        "medium_sql_gen_context": DirectSQLGenPrompts.prompt_gen_context_v0 + DirectSQLGenPrompts.q3c_info,
        "medium_sql_gen_instructions": SbySSQLGenPrompts.medium_query_instructions_2_v2,
        ####
        "adv_sql_gen_prompt_format": SbySSQLGenPrompts.adv_sql_gen_prompt_v0,
        "adv_sql_gen_task": DirectSQLGenPrompts.prompt_gen_task_v2,
        "adv_sql_gen_context": DirectSQLGenPrompts.prompt_gen_context_v0 + DirectSQLGenPrompts.q3c_info,
        "adv_sql_gen_instructions": SbySSQLGenPrompts.adv_query_instructions_2_v3
    },
}


sbscot_prompt_versions = {
    "sbscot_v0": {
        "db_schema": DBSchemaPrompts.schema_all_cntxV2,
        "sql_gen_prompt_format": DirectSQLGenPrompts.prompt_direct_gen_v0,
        "sql_gen_task": DirectSQLGenPrompts.prompt_gen_task_v0,
        "sql_gen_context": DirectSQLGenPrompts.prompt_gen_context_v0 + DirectSQLGenPrompts.q3c_info,
        "sql_gen_final_instructions": DirectSQLGenPrompts.final_instructions_sql_gen_v0,
        "medium_sql_gen_prompt_format": SbySCoTPrompts.med_decomp_dir_gen_v0,
        "medium_sql_gen_task": DirectSQLGenPrompts.prompt_gen_task_v0,
        "medium_sql_gen_context": DirectSQLGenPrompts.prompt_gen_context_v0 + DirectSQLGenPrompts.q3c_info,
        "medium_sql_gen_instructions": SbySCoTPrompts.med_query_instructions_sbscot_v0,
        "adv_sql_gen_prompt_format": SbySCoTPrompts.adv_decomp_dir_gen_v0,
        "adv_sql_gen_task": DirectSQLGenPrompts.prompt_gen_task_v0,
        "adv_sql_gen_context": DirectSQLGenPrompts.prompt_gen_context_v0 + DirectSQLGenPrompts.q3c_info,
        "adv_sql_gen_instructions": SbySCoTPrompts.adv_query_instructions_sbscot_v0
    },
    "sbscot_v1": {
        "db_schema": DBSchemaPrompts.schema_all_cntxV2,
        "sql_gen_prompt_format": SbySCoTPrompts.sbs_simple_cot_v0,
        "sql_gen_task": SbySCoTPrompts.sbs_task_cot_v0,
        "sql_gen_context": SbySCoTPrompts.sbs_context_cot_v0,
        "sql_gen_final_instructions": SbySCoTPrompts.sbs_instructions_cot_v0,
        "medium_sql_gen_prompt_format": SbySCoTPrompts.sbs_medium_cot_v0,
        "medium_sql_gen_task": SbySCoTPrompts.sbs_task_cot_v0,
        "medium_sql_gen_context": SbySCoTPrompts.sbs_context_cot_v0,
        "medium_sql_gen_instructions": SbySCoTPrompts.sbs_instructions_cot_v0,
        "adv_sql_gen_prompt_format": SbySCoTPrompts.sbs_adv_cot_v0,
        "adv_sql_gen_task": SbySCoTPrompts.sbs_task_cot_v0,
        "adv_sql_gen_context": SbySCoTPrompts.sbs_context_cot_v0,
        "adv_sql_gen_instructions": SbySCoTPrompts.sbs_instructions_cot_v0
    }
}

sc_prompt_versions = {
    "sc_v0": {
        "db_schema": DBSchemaPrompts.schema_all,
        "general_task": SelfCorrectionPrompts.general_task_selfcorr_v1,
        "general_context": SelfCorrectionPrompts.general_context_selfcorr_v1,
        "final_instructions": SelfCorrectionPrompts.final_instr_selfcorr_v0,
        "timeout_prompt_format": SelfCorrectionPrompts.self_correction_timeout_prompt_v2,
        "not_exist_prompt_format": SelfCorrectionPrompts.self_correction_other_prompt_v0,
        "schema_error_prompt_format": SelfCorrectionPrompts.self_correction_other_prompt_v0
    },
    "sc_v3": {
        "db_schema": DBSchemaPrompts.schema_all_cntxV2,
        "general_task": SelfCorrectionPrompts.general_task_selfcorr_v1,
        "general_context": SelfCorrectionPrompts.general_context_selfcorr_v1,
        "final_instructions": SelfCorrectionPrompts.final_instr_selfcorr_v0,
        "timeout_prompt_format": SelfCorrectionPrompts.self_correction_timeout_prompt_v2,
        "not_exist_prompt_format": SelfCorrectionPrompts.self_correction_no_exist_prompt_v2,
        "schema_error_prompt_format": SelfCorrectionPrompts.self_correction_schema_prompt_v2
    },
}


# join prompts from different modules
prompt_versions = {
    **schema_linking_prompt_versions,
    **diff_classification_prompt_versions,
    **dir_prompt_versions,
    **sbs_prompt_versions,
    **sbscot_prompt_versions,
    **sc_prompt_versions
}
# decomp_v5 = {"prompt_func" : base_prompt, "general_task" : simple_query_task_v2, "general_context" : simple_query_cntx + q3c_info,
#              "final_instructions" : simple_query_instructions_v2, "schema":schema_all_cntxV2_indx, "schema_decomp": schema_all_cntxV2_indx,
#              "medium_decomp_prompt": medium_decomp_prompt, "medium_decomp_task": medium_decomp_task_v3 + gpt4turbo1106_decomposed_prompt_2, "medium_query_cntx": medium_query_cntx + q3c_info, "medium_query_instructions_1": medium_query_instructions_1_v2,
#              "medium_decomp_gen": medium_decomp_gen, "medium_query_task": medium_query_task_v2, "medium_query_instructions_2": medium_query_instructions_2_v2,
#              "adv_decomp_prompt": adv_decomp_prompt, "adv_decomp_task": adv_decomp_task_v3 + gpt4turbo1106_decomposed_prompt_2, "adv_query_cntx": adv_query_cntx + q3c_info, "adv_query_instructions_1": adv_query_instructions_1_v3, 
#              "adv_decomp_gen": adv_decomp_gen, "adv_query_task": adv_query_task_v2, "adv_query_instructions_2": adv_query_instructions_2_v3}
