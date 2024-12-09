const carouselItems = document.querySelector('.carousel-items');
const carouselItem = document.querySelectorAll('.carousel-item');
const prevButton = document.querySelector('.control.left');
const nextButton = document.querySelector('.control.right');

let currentIndex = 0;
const totalItems = carouselItem.length;
const intervalTime = 4000; // 5 seconds

// Function to shift to a specific image
function shiftCarousel() {
    carouselItems.style.transition = 'transform 0.5s ease-in-out';
    carouselItems.style.transform = `translateX(-${currentIndex * 100}%)`;
}

// Move to the next image
nextButton.addEventListener('click', () => {
    if (currentIndex === totalItems - 1) {
        // Jump to the first image directly
        carouselItems.style.transition = 'none'; // Remove animation
        currentIndex = 0; // Reset index
        carouselItems.style.transform = `translateX(-${currentIndex * 100}%)`;

        // Delay re-enabling animation for smooth behavior
        setTimeout(() => {
            carouselItems.style.transition = 'transform 0.5s ease-in-out';
        });
    } else {
        currentIndex++;
        shiftCarousel();
    }
});

// Move to the previous image
prevButton.addEventListener('click', () => {
    if (currentIndex === 0) {
        // Jump to the last image directly
        carouselItems.style.transition = 'none'; // Remove animation
        currentIndex = totalItems - 1; // Go to the last image
        carouselItems.style.transform = `translateX(-${currentIndex * 100}%)`;

        // Delay re-enabling animation for smooth behavior
        setTimeout(() => {
            carouselItems.style.transition = 'transform 0.5s ease-in-out';
        });
    } else {
        currentIndex--;
        shiftCarousel();
    }
});

// Automatically shift images every 5 seconds
setInterval(() => {
    nextButton.click();
}, intervalTime);

// Initial setup
shiftCarousel();
