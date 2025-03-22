import csv
import ast

def load_recipes(filename):
    recipes = []
    with open(filename, newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            nutrition_list = ast.literal_eval(row["nutrition"])
            tags = ast.literal_eval(row["tags"])
            steps = ast.literal_eval(row["steps"])
            recipes.append({
                "name": row["name"],
                "ingredients": row["ingredients"].split(", "), 
                "n_ingredients":row["n_ingredients"],
                "minutes": row["minutes"],
                "n_steps":row["n_steps"],
                "tags":tags,
                "steps":steps,
                "description":row["description"],
                "nutrition":{
                    "Calories": nutrition_list[0], 
                    "fat": nutrition_list[1],
                    "carbohydrates": nutrition_list[2],
                    "protein": nutrition_list[3],
                    "fiber": nutrition_list[4],
                    "sugar": nutrition_list[5],
                    "sodium": nutrition_list[6],
                },

            })

    for recipe in recipes[:5]:
        print(recipe["ingredients"])
    
    return recipes