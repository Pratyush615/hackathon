from flask import Flask, render_template, jsonify, request, redirect, url_for
from data_loader import load_recipes
import recipe_recommender
import os
import tempfile
from werkzeug.utils import secure_filename
#from ingredient_detector_ai import IngredientDetector
import config as c
import json

app = Flask(__name__)

# Load all recipes
recipes = load_recipes(c.DATASET_PATH)

# Initialize recipe recommender
recommender = recipe_recommender.RecipeRecommender(recipes)

# Uncomment this when you have the model file ready
#ingredient_detector = IngredientDetector(model_path=c.MODEL_PATH)

# For demo purposes, this mock detector returns preset values
class MockIngredientDetector:
    def detect_ingredients(self, image_file):
        # Save the uploaded file temporarily to simulate processing
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, secure_filename(image_file.filename))
        image_file.save(temp_path)
        
        # Mock detection based on filename - in a real app, this would use AI
        filename = image_file.filename.lower()
        
        if 'vegetable' in filename or 'veg' in filename:
            return ['broccoli', 'carrots', 'onions', 'bell peppers']
        elif 'meat' in filename or 'protein' in filename:
            return ['chicken', 'beef', 'eggs']
        elif 'fruit' in filename:
            return ['apples', 'bananas', 'berries']
        elif 'pantry' in filename:
            return ['rice', 'pasta', 'beans', 'olive oil']
        else:
            # Default ingredients for demo
            return ['chicken', 'tomatoes', 'onions', 'garlic', 'olive oil']

# Use mock detector for now
ingredient_detector = MockIngredientDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recipe/<recipe_name>')
def show_recipe(recipe_name):
    # Find the recipe with the given name
    recipe = next((r for r in recipes if r['name'] == recipe_name), None)
    
    if not recipe:
        return redirect(url_for('index'))
    
    return render_template('recipe.html', recipe=recipe)

@app.route('/api/recipes', methods=['GET'])
def get_recipes():
    # Get filter parameters
    low_calorie = request.args.get('lowCalorie') == 'true'
    high_protein = request.args.get('highProtein') == 'true'
    high_fiber = request.args.get('highFiber') == 'true'
    low_sugar = request.args.get('lowSugar') == 'true'
    low_fat = request.args.get('lowFat') == 'true'
    quick_prep = request.args.get('quickPrep') == 'true'
    
    # Start with all recipes
    filtered_recipes = recipes
    
    # Apply filters if specified
    if low_calorie:
        filtered_recipes = [r for r in filtered_recipes if r['nutrition']['Calories'] < 500]
    
    if high_protein:
        filtered_recipes = [r for r in filtered_recipes if r['nutrition']['protein'] > 15]
    
    if high_fiber:
        filtered_recipes = [r for r in filtered_recipes if r['nutrition']['fiber'] > 5]
    
    if low_sugar:
        filtered_recipes = [r for r in filtered_recipes if r['nutrition']['sugar'] < 10]
    
    if low_fat:
        filtered_recipes = [r for r in filtered_recipes if r['nutrition']['fat'] < 15]
    
    if quick_prep:
        filtered_recipes = [r for r in filtered_recipes if int(r['minutes']) < 30]
    
    # Limit to 50 recipes for better performance
    return jsonify(filtered_recipes[:50])

@app.route('/api/detect_ingredients', methods=['POST'])
def detect_ingredients():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image = request.files['image']
    
    if image.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Detect ingredients from image
    detected_ingredients = ingredient_detector.detect_ingredients(image)
    
    return jsonify({'ingredients': detected_ingredients})

@app.route('/api/find_recipes', methods=['POST'])
def find_recipes_with_ingredients():
    data = request.json
    
    if not data or 'ingredients' not in data or not data['ingredients']:
        return jsonify({'error': 'No ingredients provided'}), 400
    
    ingredients = data['ingredients']
    
    # Use the recipe recommender to find matching recipes
    recommender.set_reqs(ingredients)
    matching_recipes = recommender.filter_recipes()
    
    # Check dietary preferences if they exist in the request
    vegetarian = data.get('vegetarian', False)
    vegan = data.get('vegan', False)
    gluten_free = data.get('glutenFree', False)
    
    if matching_recipes and (vegetarian or vegan or gluten_free):
        filtered_recipes = []
        
        for recipe in matching_recipes:
            tags = [tag.lower() for tag in recipe.get('tags', [])]
            
            if vegetarian and 'vegetarian' not in tags:
                continue
                
            if vegan and 'vegan' not in tags:
                continue
                
            if gluten_free and 'gluten-free' not in tags:
                continue
                
            filtered_recipes.append(recipe)
        
        matching_recipes = filtered_recipes
    
    if not matching_recipes:
        return jsonify([])
    
    # Limit to 50 recipes for better performance
    return jsonify(matching_recipes[:50])

@app.route('/api/popular_recipes', methods=['GET'])
def get_popular_recipes():
    # In a real app, this would be based on user ratings or views
    # For now, just return a random selection
    import random
    popular_recipes = random.sample(recipes, min(10, len(recipes)))
    return jsonify(popular_recipes)

if __name__ == '__main__':
    app.run(debug=True)