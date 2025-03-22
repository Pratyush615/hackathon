class RecipeRecommender:
    def __init__(self, dataset):
        self.dataset = dataset
        self.ingredients = []
        self.min_match = 1
        self.dietary_preferences = []

    def set_reqs(self, ingredients, min_match=1, dietary_preferences=None):
        """
        Set the requirements for recipe recommendations
        
        Args:
            ingredients (list): List of ingredients to match
            min_match (int): Minimum number of ingredients that must match
            dietary_preferences (list): Optional list of dietary preferences (vegetarian, vegan, etc.)
        """
        self.ingredients = [ing.lower() for ing in ingredients]
        self.min_match = min_match
        self.dietary_preferences = dietary_preferences or []

    def filter_recipes(self):
        """
        Filter recipes based on ingredients and dietary preferences
        
        Returns:
            list: List of matching recipes or None if no matches
        """
        filtered_recipes = []
        
        for recipe in self.dataset:
            # Check for ingredient matches
            ingredient_list = [ing.lower() for ing in recipe.get("ingredients", [])]
            
            # Count exact and partial matches
            exact_matches = sum(ing in ingredient_list for ing in self.ingredients)
            
            # Count partial matches (ingredient is part of a recipe ingredient)
            partial_matches = 0
            for user_ing in self.ingredients:
                for recipe_ing in ingredient_list:
                    if user_ing in recipe_ing and user_ing != recipe_ing:
                        partial_matches += 0.5  # Give partial weight to partial matches
                        break
            
            total_matches = exact_matches + partial_matches
            
            # Check if recipe meets the minimum match requirement
            if total_matches >= self.min_match:
                # Calculate match score (percentage of user ingredients used)
                match_percentage = (total_matches / len(self.ingredients)) * 100
                
                # Add match score to recipe
                recipe_copy = recipe.copy()
                recipe_copy['match_score'] = match_percentage
                filtered_recipes.append(recipe_copy)
        
        # Sort by match score (highest first)
        filtered_recipes.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        
        return filtered_recipes if filtered_recipes else None

    def recommend_healthy_recipes(self, max_calories=None, min_protein=None, max_fat=None):
        """
        Recommend recipes that match ingredients and meet health criteria
        
        Args:
            max_calories (int): Maximum calories per serving
            min_protein (int): Minimum protein grams per serving
            max_fat (int): Maximum fat grams per serving
            
        Returns:
            list: List of healthy recipes that match criteria
        """
        filtered_recipes = self.filter_recipes() or []
        
        # Apply health filters
        healthy_recipes = []
        
        for recipe in filtered_recipes:
            nutrition = recipe.get('nutrition', {})
            
            # Skip if doesn't meet health criteria
            if max_calories and nutrition.get('Calories', 1000) > max_calories:
                continue
                
            if min_protein and nutrition.get('protein', 0) < min_protein:
                continue
                
            if max_fat and nutrition.get('fat', 100) > max_fat:
                continue
                
            # Add health score
            health_score = self._calculate_health_score(recipe)
            recipe['health_score'] = health_score
            
            healthy_recipes.append(recipe)
        
        # Sort by health score (highest first)
        healthy_recipes.sort(key=lambda x: x.get('health_score', 0), reverse=True)
        
        return healthy_recipes if healthy_recipes else None

    def _calculate_health_score(self, recipe):
        """
        Calculate a health score for a recipe based on nutrition values
        
        Args:
            recipe (dict): Recipe data
            
        Returns:
            float: Health score between 0-100
        """
        nutrition = recipe.get('nutrition', {})
        
        score = 0
        
        # Lower calories is better (0-30 points)
        calories = nutrition.get('Calories', 500)
        if calories < 300:
            score += 30
        elif calories < 500:
            score += 20
        elif calories < 700:
            score += 10
        
        # Higher protein is better (0-25 points)
        protein = nutrition.get('protein', 10)
        if protein > 30:
            score += 25
        elif protein > 20:
            score += 20
        elif protein > 10:
            score += 15
        
        # Higher fiber is better (0-25 points)
        fiber = nutrition.get('fiber', 3)
        if fiber > 10:
            score += 25
        elif fiber > 5:
            score += 15
        elif fiber > 3:
            score += 10
        
        # Lower sugar is better (0-20 points)
        sugar = nutrition.get('sugar', 10)
        if sugar < 5:
            score += 20
        elif sugar < 10:
            score += 15
        elif sugar < 20:
            score += 5
        
        return score

    def recommend(self, healthy_only=False):
        """
        Main recommendation function
        
        Args:
            healthy_only (bool): Whether to return only healthy recipes
            
        Returns:
            list: Recommended recipes
        """
        if healthy_only:
            return self.recommend_healthy_recipes() or "No matching healthy recipes found"
        else:
            return self.filter_recipes() or "No matching recipes found"