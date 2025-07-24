console.log("Script loaded successfully!!!!!!!");

let jsonData = fetch('/static/data.json')

  .then(response => response.json())

  .then(data => {

    console.log("Data fetched successfully:", data);

  })

  .catch(error => {

    console.error("Error fetching data:", error);

  });