var removeCards = document.getElementsByClassName('reduire');

for (var i = 0; i < removeCards.length; i++){
    var button = removeCards[i]; 
    button.addEventListener('click', function(event){
        var buttonClicked = event.target;
        if (buttonClicked.innerHTML == "réduire"){
            buttonClicked.parentElement.lastElementChild.style.display = "None";
            buttonClicked.parentElement.style.height = "60px";
            buttonClicked.innerHTML = "agrandir";
        }
        
        else if (buttonClicked.innerHTML == "agrandir"){
            buttonClicked.parentElement.style.height = "auto";
            buttonClicked.parentElement.lastElementChild.style.display = "block";
            buttonClicked.innerHTML = "réduire";
        }

    })
} 