document.addEventListener("DOMContentLoaded", function () {
  console.log("Rain-to-Road application initialized successfully.");
  
  // Dynamic Route Distance Estimator (Interactive Form UX)
  const originSelect = document.getElementById("origin");
  const destSelect = document.getElementById("destination");
  const distanceInput = document.getElementById("distance");

  if (originSelect && destSelect && distanceInput) {
    function updateDistance() {
      if (originSelect.value === destSelect.value) {
        distanceInput.value = "2.5";
      } else {
        const hash = (originSelect.value.length + destSelect.value.length) % 15;
        distanceInput.value = (3.5 + hash * 0.8).toFixed(1);
      }
    }

    originSelect.addEventListener("change", updateDistance);
    destSelect.addEventListener("change", updateDistance);
  }
});