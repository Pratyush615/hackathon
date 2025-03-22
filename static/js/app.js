$(document).ready(function() {
    // Global variables
    let allRecipes = [];
    let selectedIngredients = [];
    
    // Initial load of recipes
    loadRecipes();
    
    // Function to load recipes from API
    function loadRecipes(filters = {}) {
        // Show loading spinner
        $('#loading-spinner').show();
        $('#recipe-list').empty();
        
        $.ajax({
            url: '/api/recipes',
            type: 'GET',
            data: filters,
            success: function(data) {
                // Store all recipes globally
                allRecipes = data;
                
                // Hide loading spinner
                $('#loading-spinner').hide();
                
                // Display recipes
                displayRecipes(data);
                
                // Initialize healthy recipe filters
                initializeFilters();
            },
            error: function() {
                $('#loading-spinner').hide();
                $('#recipe-list').html('<div class="alert alert-danger">Failed to load recipes. Please try again later.</div>');
            }
        });
    }
    
    // Function to display recipes
    function displayRecipes(recipes) {
        const recipeList = $('#recipe-list');
        recipeList.empty();
        
        if (recipes.length === 0) {
            recipeList.html(`
                <div class="empty-state fade-in">
                    <i class="fas fa-utensils"></i>
                    <h3>No recipes found</h3>
                    <p>Try adjusting your filters or adding different ingredients.</p>
                </div>
            `);
            return;
        }
        
        recipes.forEach(function(recipe, index) {
            // Calculate health score based on nutrition
            const healthScore = calculateHealthScore(recipe);
            const healthIndicator = getHealthIndicator(healthScore);
            
            // Get relevant health tags
            const healthTags = getHealthTags(recipe);
            
            // Create recipe card with animation delay based on index
            const recipeItem = `
                <div class="card mb-4 fade-in" style="animation-delay: ${index * 0.1}s">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-start">
                            <h5 class="card-title">${recipe.name}</h5>
                            <span class="badge ${healthIndicator.class}">${healthIndicator.text}</span>
                        </div>
                        <div class="health-tags mb-2">
                            ${healthTags.map(tag => `<span class="badge bg-info me-1">${tag}</span>`).join('')}
                        </div>
                        <p><strong>Ingredients:</strong> ${recipe.ingredients.join(', ')}</p>
                        <p><strong>Preparation Time:</strong> ${recipe.minutes} min</p>
                        <p><strong>Calories:</strong> ${recipe.nutrition.Calories}</p>
                        <a href="/recipe/${recipe.name}" class="btn btn-outline-primary btn-sm mt-2">View Details</a>
                    </div>
                </div>
            `;
            recipeList.append(recipeItem);
        });
    }
    
    // Function to calculate health score
    function calculateHealthScore(recipe) {
        // Simple health score calculation
        let score = 0;
        
        // Lower calories is better (0-30 points)
        if (recipe.nutrition.Calories < 300) score += 30;
        else if (recipe.nutrition.Calories < 500) score += 20;
        else if (recipe.nutrition.Calories < 700) score += 10;
        
        // Higher protein is better (0-25 points)
        if (recipe.nutrition.protein > 30) score += 25;
        else if (recipe.nutrition.protein > 20) score += 20;
        else if (recipe.nutrition.protein > 10) score += 15;
        
        // Higher fiber is better (0-25 points)
        if (recipe.nutrition.fiber > 10) score += 25;
        else if (recipe.nutrition.fiber > 5) score += 15;
        else if (recipe.nutrition.fiber > 3) score += 10;
        
        // Lower sugar is better (0-20 points)
        if (recipe.nutrition.sugar < 5) score += 20;
        else if (recipe.nutrition.sugar < 10) score += 15;
        else if (recipe.nutrition.sugar < 20) score += 5;
        
        return score;
    }
    
    // Function to get health indicator based on score
    function getHealthIndicator(score) {
        if (score >= 75) {
            return { text: 'Very Healthy', class: 'bg-success' };
        } else if (score >= 50) {
            return { text: 'Healthy', class: 'bg-info' };
        } else if (score >= 25) {
            return { text: 'Moderate', class: 'bg-warning' };
        } else {
            return { text: 'Indulgent', class: 'bg-secondary' };
        }
    }
    
    // Function to get health tags
    function getHealthTags(recipe) {
        const tags = [];
        
        if (recipe.nutrition.Calories < 500) tags.push('Low Calorie');
        if (recipe.nutrition.protein > 15) tags.push('High Protein');
        if (recipe.nutrition.fiber > 5) tags.push('High Fiber');
        if (recipe.nutrition.sugar < 5) tags.push('Low Sugar');
        if (recipe.nutrition.fat < 10) tags.push('Low Fat');
        
        // Limit to 3 tags max
        return tags.slice(0, 3);
    }
    
    // Initialize filter controls
    function initializeFilters() {
        // Create healthy recipe filter options
        const filterSection = `
            <div class="filters-section mb-4 fade-in">
                <h3>Filter Recipes</h3>
                <div class="row mt-3">
                    <div class="col-md-4">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="low-calorie">
                            <label class="form-check-label" for="low-calorie">Low Calorie</label>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="high-protein">
                            <label class="form-check-label" for="high-protein">High Protein</label>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="high-fiber">
                            <label class="form-check-label" for="high-fiber">High Fiber</label>
                        </div>
                    </div>
                </div>
                <div class="row mt-2">
                    <div class="col-md-4">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="low-sugar">
                            <label class="form-check-label" for="low-sugar">Low Sugar</label>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="low-fat">
                            <label class="form-check-label" for="low-fat">Low Fat</label>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="quick-prep">
                            <label class="form-check-label" for="quick-prep">Quick Prep (<30 min)</label>
                        </div>
                    </div>
                </div>
                <div class="mt-3">
                    <button id="apply-filters" class="btn btn-primary me-2">Apply Filters</button>
                    <button id="reset-filters" class="btn btn-outline-secondary">Reset</button>
                </div>
            </div>
        `;
        
        // Add filter section before recipe list
        $('#recipe-list').before(filterSection);
        
        // Handle filter button clicks
        $('#apply-filters').click(function() {
            applyFilters();
        });
        
        $('#reset-filters').click(function() {
            resetFilters();
        });
    }
    
    // Apply filters to recipe list
    function applyFilters() {
        const filters = {
            lowCalorie: $('#low-calorie').is(':checked'),
            highProtein: $('#high-protein').is(':checked'),
            highFiber: $('#high-fiber').is(':checked'),
            lowSugar: $('#low-sugar').is(':checked'),
            lowFat: $('#low-fat').is(':checked'),
            quickPrep: $('#quick-prep').is(':checked')
        };
        
        // Filter recipes based on criteria
        const filteredRecipes = allRecipes.filter(recipe => {
            if (filters.lowCalorie && recipe.nutrition.Calories >= 500) return false;
            if (filters.highProtein && recipe.nutrition.protein <= 15) return false;
            if (filters.highFiber && recipe.nutrition.fiber <= 5) return false;
            if (filters.lowSugar && recipe.nutrition.sugar >= 10) return false;
            if (filters.lowFat && recipe.nutrition.fat >= 15) return false;
            if (filters.quickPrep && parseInt(recipe.minutes) >= 30) return false;
            
            // If ingredients are selected, filter by those too
            if (selectedIngredients.length > 0) {
                const matchingIngredients = selectedIngredients.filter(ing => 
                    recipe.ingredients.some(recipeIng => 
                        recipeIng.toLowerCase().includes(ing.toLowerCase())
                    )
                );
                if (matchingIngredients.length === 0) return false;
            }
            
            return true;
        });
        
        // Display filtered recipes with animation
        displayRecipes(filteredRecipes);
    }
    
    // Reset all filters
    function resetFilters() {
        // Uncheck all filter checkboxes
        $('.form-check-input').prop('checked', false);
        
        // Clear selected ingredients
        selectedIngredients = [];
        updateSelectedIngredientsDisplay();
        
        // Reload all recipes
        displayRecipes(allRecipes);
    }

    // Handle image upload for ingredient detection
    $('#image-form').submit(function(e) {
        e.preventDefault();

        const formData = new FormData();
        const fileInput = $('#image-upload')[0];
        
        if (fileInput.files.length === 0) {
            alert('Please select an image first');
            return;
        }
        
        formData.append('image', fileInput.files[0]);

        // Show loading state
        $('#detected-ingredients').html('<div class="loading-spinner"></div>');
        $('.loading-spinner').show();

        $.ajax({
            url: '/api/detect_ingredients',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $('.loading-spinner').hide();
                
                if (!response.ingredients || response.ingredients.length === 0) {
                    $('#detected-ingredients').html('<p>No ingredients detected. Try another image.</p>');
                    return;
                }
                
                const ingredientsList = $('#detected-ingredients');
                ingredientsList.empty();
                
                // Create ingredient tags and add to selected ingredients list
                response.ingredients.forEach(function(ingredient) {
                    const tag = `<span class="ingredient-tag" data-ingredient="${ingredient}">${ingredient}</span>`;
                    ingredientsList.append(tag);
                });
                
                // Initialize click events for ingredient tags
                $('.ingredient-tag').click(function() {
                    const ingredient = $(this).data('ingredient');
                    toggleIngredient(ingredient, $(this));
                });
                
                // Add recommendation section if ingredients were detected
                if (!$('#recommendation-section').length) {
                    $('#detected-ingredients').after(`
                        <div id="recommendation-section" class="mt-4 fade-in">
                            <h3>Selected Ingredients</h3>
                            <div id="selected-ingredients-list" class="mb-3"></div>
                            <button id="find-recipes" class="btn btn-primary">Find Recipes</button>
                        </div>
                    `);
                }
                
                // Handle find recipes button click
                $('#find-recipes').off('click').on('click', function() {
                    if (selectedIngredients.length === 0) {
                        alert('Please select at least one ingredient');
                        return;
                    }
                    
                    findRecipesWithIngredients();
                });
            },
            error: function() {
                $('.loading-spinner').hide();
                $('#detected-ingredients').html('<p class="text-danger">Failed to process image. Please try again.</p>');
            }
        });
    });
    
    // Toggle ingredient selection
    function toggleIngredient(ingredient, element) {
        const index = selectedIngredients.indexOf(ingredient);
        
        if (index === -1) {
            // Add ingredient
            selectedIngredients.push(ingredient);
            element.addClass('selected');
        } else {
            // Remove ingredient
            selectedIngredients.splice(index, 1);
            element.removeClass('selected');
        }
        
        updateSelectedIngredientsDisplay();
    }
    
    // Update the display of selected ingredients
    function updateSelectedIngredientsDisplay() {
        const container = $('#selected-ingredients-list');
        
        if (selectedIngredients.length === 0) {
            container.html('<p>No ingredients selected</p>');
            return;
        }
        
        container.empty();
        
        selectedIngredients.forEach(function(ingredient) {
            container.append(`
                <div class="selected-ingredient fade-in">
                    <span>${ingredient}</span>
                    <button class="btn-remove" data-ingredient="${ingredient}">×</button>
                </div>
            `);
        });
        
        // Handle remove button clicks
        $('.btn-remove').click(function() {
            const ingredient = $(this).data('ingredient');
            const index = selectedIngredients.indexOf(ingredient);
            
            if (index !== -1) {
                selectedIngredients.splice(index, 1);
                updateSelectedIngredientsDisplay();
                
                // Update tag status in detected ingredients
                $(`.ingredient-tag[data-ingredient="${ingredient}"]`).removeClass('selected');
            }
        });
    }
    
    // Find recipes with selected ingredients
    function findRecipesWithIngredients() {
        // Show loading spinner
        $('#loading-spinner').show();
        
        $.ajax({
            url: '/api/find_recipes',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ ingredients: selectedIngredients }),
            success: function(data) {
                $('#loading-spinner').hide();
                
                // Scroll to recipe list
                $('html, body').animate({
                    scrollTop: $("#recipe-list").offset().top - 100
                }, 500);
                
                displayRecipes(data);
            },
            error: function() {
                $('#loading-spinner').hide();
                alert('Failed to find recipes. Please try again.');
            }
        });
    }
    
    // Add manual ingredient entry
    $('#add-ingredient-btn').click(function() {
        const ingredient = $('#manual-ingredient').val().trim();
        
        if (ingredient === '') {
            alert('Please enter an ingredient');
            return;
        }
        
        if (selectedIngredients.indexOf(ingredient) === -1) {
            selectedIngredients.push(ingredient);
            updateSelectedIngredientsDisplay();
            $('#manual-ingredient').val('');
        }
    });
});