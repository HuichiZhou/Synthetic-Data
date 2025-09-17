import asyncio
import random
from typing import Any, Dict, List

from tqdm.asyncio import tqdm

from utils.async_gemini import AsyncGeminiClient
from prompt import ENTITY_GENERATE_PROMPT_TEMPLATE, INSTRUCTION_DIMENSIONS


def generate_instruction_combinations(
    dimensions: Dict[str, Any],
    sample_size: int = 100,
    ensure_diversity: bool = True
) -> List[Dict[str, Any]]:
    """Generate diverse instruction combinations in English"""
    
    combinations = []
    
    # Flatten nested dimensions
    flat_disciplines = []
    for category, disciplines in dimensions["disciplines"].items():
        flat_disciplines.extend(disciplines)
    
    flat_methodologies = []
    for category, methods in dimensions["methodologies"].items():
        flat_methodologies.extend(methods)
    
    for i in range(sample_size):
        # Select primary discipline
        primary_discipline = random.choice(flat_disciplines)
        
        # Select topic based on discipline
        if primary_discipline in dimensions["topics_by_discipline"]:
            topic = random.choice(dimensions["topics_by_discipline"][primary_discipline])
        else:
            # Randomly select a topic from all available
            all_topics = []
            for topics in dimensions["topics_by_discipline"].values():
                all_topics.extend(topics)
            topic = random.choice(all_topics)
        
        # Select cross-disciplinary perspectives
        perspective_count = random.choice(dimensions["openness_parameters"]["Perspective Count"])
        cross_disciplines = random.sample(
            [d for d in flat_disciplines if d != primary_discipline], 
            min(perspective_count - 1, len(flat_disciplines) - 1)
        )
        
        # Generate random attributes for the template
        all_dimension_keys = list(dimensions.keys())
        random_key1 = random.choice(all_dimension_keys)
        random_key2 = random.choice([k for k in all_dimension_keys if k != random_key1])
        
        # Get random values for the selected keys
        if isinstance(dimensions[random_key1], dict):
            random_value1 = random.choice(list(dimensions[random_key1].keys()))
        elif isinstance(dimensions[random_key1], list):
            random_value1 = random.choice(dimensions[random_key1])
        else:
            random_value1 = str(dimensions[random_key1])
        
        if isinstance(dimensions[random_key2], dict):
            random_value2 = random.choice(list(dimensions[random_key2].keys()))
        elif isinstance(dimensions[random_key2], list):
            random_value2 = random.choice(dimensions[random_key2])
        else:
            random_value2 = str(dimensions[random_key2])
        
        combination = {
            "topic": topic,
            "random_key1": random_key1.replace("_", " ").title(),
            "random_value1": random_value1,
            "random_key2": random_key2.replace("_", " ").title(),
            "random_value2": random_value2,
            "parameters": {
                "Primary Discipline": primary_discipline,
                "Cross Disciplines": ", ".join(cross_disciplines) if cross_disciplines else "None",
                "Task Type": random.choice(dimensions["task_types"]),
                "Methodology": ", ".join(random.sample(flat_methodologies, random.randint(1, 3))),
                "Geographical Scope": random.choice(dimensions["geographical_scope"]),
                "Target Population": random.choice(dimensions["target_populations"]),
                "Time Horizon": random.choice(dimensions["time_horizons"]),
                "Data Sources": ", ".join(random.sample(dimensions["data_sources"], random.randint(1, 3))),
                "Ethical Constraints": ", ".join(random.sample(dimensions["ethical_constraints"], random.randint(1, 3))),
                "Regulatory Framework": random.choice(dimensions["regulatory_frameworks"]),
                "Budget": random.choice(dimensions["resource_constraints"]["Budget"]),
                "Timeline": random.choice(dimensions["resource_constraints"]["Timeline"]),
                "Data Access": random.choice(dimensions["resource_constraints"]["Data Access"]),
                "Computing Resources": random.choice(dimensions["resource_constraints"]["Computing Resources"]),
                "Team Size": random.choice(dimensions["resource_constraints"]["Team Size"]),
                "Report Type": random.choice(dimensions["output_formats"]["Report Types"]),
                "Target Audience": random.choice(dimensions["output_formats"]["Target Audiences"]),
                "Report Length": random.choice(dimensions["output_formats"]["Report Lengths"]),
                "Writing Style": random.choice(dimensions["output_formats"]["Writing Styles"]),
                "Visualization": ", ".join(random.sample(dimensions["output_formats"]["Visualization Types"], random.randint(1, 3))),
                "Alternative Solutions Required": random.choice(dimensions["openness_parameters"]["Alternative Count"]),
                "Perspective Count": perspective_count,
                "Uncertainty Analysis": f"Identify {random.choice(dimensions['openness_parameters']['Uncertainty Count'])} key uncertainties",
                "Tradeoff Analysis": f"Analyze tradeoffs across {random.choice(dimensions['openness_parameters']['Tradeoff Dimensions'])} dimensions",
                "Scenario Planning": f"Develop {random.choice(dimensions['openness_parameters']['Scenario Count'])} scenarios",
                "Analysis Requirements": ", ".join(random.sample(dimensions["analysis_requirements"], random.randint(2, 4))),
                "Research Approach": random.choice(dimensions["research_approaches"])
            }
        }
        
        combinations.append(combination)
    
    return combinations

async def batch_deal(prompts, client):
    tasks = [ENTITY_GENERATE_PROMPT_TEMPLATE.format(**prompt) for prompt in prompts]
    response = await client.generate_batch(tasks)

    result = []
    for concept, resp in zip(tasks, response):
        result.append({"concept": concept, "response": resp})

    return result


if __name__ == "__main__":
    combinations = generate_instruction_combinations(
        INSTRUCTION_DIMENSIONS, 
        sample_size=200
    )

    chosen = asyncio.run(batch_deal(combinations, AsyncGeminiClient(max_concurrent=80, model="gemini-2.5-pro")))

    with open("result/generated_entities.jsonl", "w") as f:
        for item in tqdm(chosen):
            f.write(f"{item}\n")