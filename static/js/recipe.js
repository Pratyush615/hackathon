$(document).ready(function() {
    // Add fade-in animation to recipe details
    $('.recipe-details').addClass('fade-in');
    
    // Animate nutrition items with slight delay
    $('.nutrition-item').each(function(index) {
        $(this).css('animation-delay', `${index * 0.1}s`);
        $(this).addClass('fade-in');
    });
    
    // Make ingredients clickable to add to shopping list
    let shoppingList = [];
    
    $('.ingredients-list li').click(function() {
        const ingredient = $(this).text().trim();
        
        // Toggle active class
        $(this).toggleClass('active');
        
        if ($(this).hasClass('active')) {
            // Add to shopping list
            if (!shoppingList.includes(ingredient)) {
                shoppingList.push(ingredient);
                
                // Show shopping list if first item
                if (shoppingList.length === 1) {
                    addShoppingListUI();
                }
                
                updateShoppingListUI();
            }
        } else {
            // Remove from shopping list
            const index = shoppingList.indexOf(ingredient);
            if (index !== -1) {
                shoppingList.splice(index, 1);
                updateShoppingListUI();
                
                // Hide shopping list if empty
                if (shoppingList.length === 0) {
                    $('#shopping-list-container').remove();
                }
            }
        }
    });
    
    // Function to add shopping list UI
    function addShoppingListUI() {
        $('<div id="shopping-list-container" class="card mt-4 fade-in">' +
            '<div class="card-body">' +
                '<h3>Shopping List</h3>' +
                '<ul id="shopping-list"></ul>' +
                '<button id="clear-list" class="btn btn-outline-danger btn-sm mt-2">Clear List</button>' +
            '</div>' +
          '</div>').insertAfter('.recipe-details');
        
        // Handle clear button
        $('#clear-list').click(function() {
            shoppingList = [];
            $('.ingredients-list li').removeClass('active');
            $('#shopping-list-container').slideUp(300, function() {
                $(this).remove();
            });
        });
    }
    
    // Function to update shopping list UI
    function updateShoppingListUI() {
        const list = $('#shopping-list');
        list.empty();
        
        shoppingList.forEach(function(item) {
            list.append(`<li>${item} <button class="btn-remove" data-item="${item}">×</button></li>`);
        });
        
        // Handle remove buttons
        $('.btn-remove').click(function() {
            const item = $(this).data('item');
            const index = shoppingList.indexOf(item);
            
            if (index !== -1) {
                shoppingList.splice(index, 1);
                
                // Remove active class from ingredient list
                $(`.ingredients-list li:contains('${item}')`).removeClass('active');
                
                updateShoppingListUI();
                
                // Hide shopping list if empty
                if (shoppingList.length === 0) {
                    $('#shopping-list-container').remove();
                }
            }
        });
    }
    
    // Add print recipe button
    $('<button id="print-recipe" class="btn btn-outline-primary mt-3">' +
        '<i class="fas fa-print me-2"></i>Print Recipe' +
      '</button>').appendTo('.recipe-details .card-body');
    
    $('#print-recipe').click(function() {
        window.print();
    });
});