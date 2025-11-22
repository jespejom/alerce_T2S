import json

# from pipeline.process import *
# from prompts.base.prompts import *
from prompts import DBSchemaPrompts, SchemaLinkingPrompts
from prompts import (
    DBSchemaPrompts, SchemaLinkingPrompts, DiffClassificationPrompts, Q3cClassificationPrompts, SelfCorrectionPrompts,
    DirectSQLGenPrompts
)
from prompts.SbySPrompts import (
    SbySPlanPrompts, SbySSQLGenPrompts, SbySCoTPrompts
)
from prompts.prompts_pipeline import *

# Prompt dictionary guideline and used by Jorge
prompts = {
    "Schema Linking": {
        "base_prompt": schema_linking_prompt_versions["sl_v3"]["schema_linking_format"].format(db_schema=schema_linking_prompt_versions["sl_v3"]["db_schema"],
                                                                                                final_instructions=schema_linking_prompt_versions["sl_v3"]["sl_final_instructions"]),
        "context1": DBSchemaPrompts.schema_all_cntxV2,
        "context2": DBSchemaPrompts.schema_all_cntxV2_indx,
        "context3": DBSchemaPrompts.schema_all_cntxV2_indx,
    },
    "Classify": {
        # "base_prompt": diff_class_prompt_v7,
        "base_prompt": diff_classification_prompt_versions["diff_v8"]["diff_class_format"],
        # "final_instructions": final_instructions_diff_v2
        "final_instructions": diff_classification_prompt_versions["diff_v8"]["diff_final_instructions"]
    },
    "Decomposition": {
        "simple": {
            "base_prompt": sbs_prompt_versions["sbs_v4"]["sql_gen_prompt_format"],
            "query_task": sbs_prompt_versions["sbs_v4"]["sql_gen_task"],
            "query_context": sbs_prompt_versions["sbs_v4"]["sql_gen_context"],
            "external_knowledge": "placeholder",
            "domain_knowledge": "placeholder",
            "query_instructions": sbs_prompt_versions["sbs_v4"]["sql_gen_final_instructions"]
        },
        "medium": {
            "decomp_plan": {
                "base_prompt": sbs_prompt_versions["sbs_v4"]["medium_plan_prompt_format"],
                "decomp_task": sbs_prompt_versions["sbs_v4"]["medium_plan_task"],
                # "decomp_task_python": medium_decomp_task_v3 + gpt4turbo1106_decomposed_prompt_2_python,
                "query_context": sbs_prompt_versions["sbs_v4"]["medium_plan_context"],
                "query_instructions": sbs_prompt_versions["sbs_v4"]["medium_plan_instructions"]
            },
            "decomp_gen": {
                "sql": {
                    "base_prompt": sbs_prompt_versions["sbs_v4"]["medium_sql_gen_prompt_format"],
                    "query_task": sbs_prompt_versions["sbs_v4"]["medium_sql_gen_task"],
                    "query_instructions": sbs_prompt_versions["sbs_v4"]["medium_sql_gen_instructions"],
                },
                # "python": {
                #     "base_prompt": medium_decomp_gen_vf_python,
                #     "query_task": medium_query_task_v2,
                #     "query_instructions": medium_query_instructions_2_v2_python,
                # }
            }
        },
        "advanced": {
            "decomp_plan": {
                "base_prompt": sbs_prompt_versions["sbs_v4"]["adv_plan_prompt_format"],
                "decomp_task": sbs_prompt_versions["sbs_v4"]["adv_plan_task"],
                # "decomp_task_python": sbs_prompt_versions["sbs_v4"]["adv_decomp_task_python"],
                "query_context": sbs_prompt_versions["sbs_v4"]["adv_plan_context"],
                "query_instructions": sbs_prompt_versions["sbs_v4"]["adv_plan_instructions"]
            },
            "decomp_gen": {
                "sql": {
                    "base_prompt": sbs_prompt_versions["sbs_v4"]["adv_sql_gen_prompt_format"],
                    "query_task": sbs_prompt_versions["sbs_v4"]["adv_sql_gen_task"],
                    "query_instructions": sbs_prompt_versions["sbs_v4"]["adv_sql_gen_instructions"],
                },
                # "python": {
                #     "base_prompt": adv_decomp_gen_vf_python,
                #     "query_task": adv_query_task_v2,
                #     "query_instructions": adv_query_instructions_2_v3_python,
                # }
            }
        }
    },
    "Direct": {
        "base_prompt": {
            "base_prompt": dir_prompt_versions["dir_v8"]["sql_gen_prompt_format"],
            "general_task": dir_prompt_versions["dir_v8"]["sql_gen_task"],
            "general_context": dir_prompt_versions["dir_v8"]["sql_gen_context"],
            "final_instructions": dir_prompt_versions["dir_v8"]["sql_gen_final_instructions"]
        },
        "request_prompt": {
            "external_knowledge": "placeholder",
            "domain_knowledge": "placeholder"
        }
    }
}

# Write the dictionary to a JSON file
# with open("final_prompts/prompts_v4.json", "w", encoding="utf-8") as f:
#     json.dump(prompts, f, ensure_ascii=False, indent=4)

with open("prompts_jorge.json", "w", encoding="utf-8") as f:
    json.dump(prompts, f, ensure_ascii=False, indent=4)