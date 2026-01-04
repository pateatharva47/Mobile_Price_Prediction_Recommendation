// Main JavaScript file for Mobile Price Predictor

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initializeApp();
});

function initializeApp() {
    // Initialize smooth scrolling
    initSmoothScrolling();
    
    // Initialize forms
    initPredictionForm();
    initRecommendationForm();
    
    // Initialize animations
    initScrollAnimations();
    
    // Initialize tooltips
    initTooltips();
}

// Smooth scrolling for navigation links
function initSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const offsetTop = target.offsetTop - 80; // Account for fixed navbar
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Initialize prediction form
function initPredictionForm() {
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('prediction-result');
    const priceSpan = document.getElementById('predicted-price');
    
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Validate form first
            const formData = new FormData(form);
            if (!validateForm(formData)) {
                return;
            }
            
            // Get form data
            const data = {
                brand: formData.get('brand'),
                operating_system: formData.get('operating_system'),
                release_year: formData.get('release_year'),
                screen_size: formData.get('screen_size'),
                ram: formData.get('ram'),
                storage: formData.get('storage'),
                battery: formData.get('battery'),
                processor: formData.get('processor')
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const result = await response.json();
                
                if (result.success) {
                    // Format and display price instantly
                    const formattedPrice = formatIndianPrice(result.predicted_price);
                    priceSpan.textContent = formattedPrice;
                    
                    // Show result instantly
                    resultDiv.style.display = 'block';
                    resultDiv.classList.add('fade-in');
                    
                    // Quick scroll to result
                    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    
                    // Show success notification
                    showNotification('Price predicted successfully!', 'success');
                } else {
                    showNotification('Error: ' + (result.error || 'Unknown error'), 'error');
                }
            } catch(error) {
                showNotification('Network error. Please try again.','error');
            }
        });
    }
}

// Initialize recommendation form
function initRecommendationForm() {
    const form = document.getElementById('recommendation-form');
    const resultDiv = document.getElementById('recommendations-result');
    const listDiv = document.getElementById('recommendations-list');
    
    if (form) {
        form.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // Get form data
            const formData = new FormData(form);
            const data = {
                budget_min: formData.get('budget_min') || 0,
                budget_max: formData.get('budget_max') || 100000,
                brand: formData.get('brand') || '',
                operating_system: formData.get('operating_system') || '',
                ram: formData.get('ram') || 0,
                storage: formData.get('storage') || 0,
                battery: formData.get('battery') || 0
            };
            
            try {
                const response = await fetch('/recommend', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const result = await response.json();
                
                if (result.success) {
                    displayMobileRecommendations(result.recommendations, listDiv);
                    
                    // Show result instantly
                    resultDiv.style.display = 'block';
                    resultDiv.classList.add('fade-in');
                    
                    // Quick scroll to result
                    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    
                    // Show success notification
                    showNotification(`Found ${result.recommendations.length} mobile recommendations!`, 'success');
                } else {
                    showNotification('Error: ' + (result.error || 'Unknown error'), 'error');
                }
            } catch(error){
                showNotification('Network error. Please try again.','error');
            }
        });
    }
}

// Display mobile recommendations
function displayMobileRecommendations(recommendations, container) {
    container.innerHTML = '';
    
    if (recommendations.length === 0) {
        container.innerHTML = `
            <div class="text-center py-4">
                <i class="fas fa-mobile-alt fa-3x text-muted mb-3"></i>
                <h5>No mobiles found</h5>
                <p class="text-muted">Try adjusting your budget or requirements to find more options.</p>
            </div>
        `;
        return;
    }
    
    recommendations.forEach((mobile, index) => {
        const item = document.createElement('div');
        item.className = 'mobile-recommendation-item';
        item.style.animationDelay = `${index * 0.1}s`;
        
        item.innerHTML = `
            <div class="mobile-card">
                <div class="mobile-header">
                    <h5><i class="fas fa-mobile-alt me-2"></i>${mobile.recommended_item}</h5>
                </div>
                <div class="mobile-specs">
                    <p><i class="fas fa-cogs me-2"></i>${mobile.specifications}</p>
                </div>
                <div class="mobile-footer">
                    <div class="price-info">
                        <strong>${mobile.price}</strong>
                        ${mobile.price_range ? `<br><small class="text-muted">${mobile.price_range}</small>` : ''}
                    </div>
                    <div class="mobile-badges">
                        ${mobile.brand ? `<span class="badge bg-primary">${mobile.brand}</span>` : ''}
                        ${mobile.os ? `<span class="badge bg-secondary">${mobile.os}</span>` : ''}
                    </div>
                </div>
            </div>
        `;
        
        container.appendChild(item);
    });
}

// Format price in Indian number system
function formatIndianPrice(price) {
    const numStr = Math.round(price).toString();
    const lastThree = numStr.substring(numStr.length - 3);
    const otherNumbers = numStr.substring(0, numStr.length - 3);
    
    if (otherNumbers) {
        return otherNumbers.replace(/\B(?=(\d{2})+(?!\d))/g, ",") + "," + lastThree;
    } else {
        return lastThree;
    }
}

// Show notification
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type === 'error' ? 'danger' : 'success'} alert-dismissible fade show notification`;
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        z-index: 9999;
        min-width: 300px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        border: none;
        border-radius: 10px;
    `;
    
    notification.innerHTML = `
        <i class="fas fa-${type === 'error' ? 'exclamation-triangle' : 'check-circle'} me-2"></i>
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 3000);
}

// Initialize scroll animations
function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('fade-in');
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    document.querySelectorAll('.feature-card, .prediction-card, .recommendation-card').forEach(el => {
        observer.observe(el);
    });
}

// Initialize tooltips
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Navbar scroll effect
window.addEventListener('scroll', function() {
    const navbar = document.querySelector('.navbar');
    if (window.scrollY > 50) {
        navbar.style.background = 'linear-gradient(135deg, rgba(102, 126, 234, 0.95), rgba(118, 75, 162, 0.95))';
        navbar.style.backdropFilter = 'blur(10px)';
    } else {
        navbar.style.background = 'linear-gradient(135deg, var(--primary-color), var(--secondary-color))';
        navbar.style.backdropFilter = 'none';
    }
});

// Form validation helpers
function validateForm(formData) {
    const required = ['brand', 'operating_system', 'release_year', 'screen_size', 'ram', 'storage', 'battery', 'processor'];
    
    for (let field of required) {
        const value = formData.get(field);
        if (!value || value.trim() === '') {
            const fieldName = field.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
            showNotification(`Please select/enter ${fieldName}.`, 'error');
            
            // Focus on the field
            const fieldElement = document.querySelector(`[name="${field}"]`);
            if (fieldElement) {
                fieldElement.focus();
                fieldElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            return false;
        }
    }
    
    // Validate numeric ranges
    const screenSize = parseFloat(formData.get('screen_size'));
    const battery = parseInt(formData.get('battery'));
    
    if (isNaN(screenSize) || screenSize < 4 || screenSize > 8) {
        showNotification('Screen size should be between 4 and 8 inches.', 'error');
        document.querySelector('[name="screen_size"]').focus();
        return false;
    }
    
    if (isNaN(battery) || battery < 2000 || battery > 6000) {
        showNotification('Battery capacity should be between 2000 and 6000 mAh.', 'error');
        document.querySelector('[name="battery"]').focus();
        return false;
    }
    
    return true;
}

// Add keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to submit forms
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const activeForm = document.activeElement.closest('form');
        if (activeForm) {
            activeForm.dispatchEvent(new Event('submit'));
        }
    }
});

// Add form auto-save (optional)
function initAutoSave() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        const inputs = form.querySelectorAll('input, select');
        
        inputs.forEach(input => {
            input.addEventListener('change', function() {
                const formId = form.id;
                const inputName = this.name;
                const inputValue = this.value;
                
                // Save to localStorage
                localStorage.setItem(`${formId}_${inputName}`, inputValue);
            });
        });
        
        // Restore saved values
        const inputs2 = form.querySelectorAll('input, select');
        inputs2.forEach(input => {
            const formId = form.id;
            const inputName = input.name;
            const savedValue = localStorage.getItem(`${formId}_${inputName}`);
            
            if (savedValue) {
                input.value = savedValue;
            }
        });
    });
}

// Initialize auto-save
// initAutoSave(); // Uncomment if you want auto-save functionality