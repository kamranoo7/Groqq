const answers = [];
let questions = null;
const submitButton = document.querySelector(".submit");
const container = document.querySelector(".slider-slides");
let count = 1;
const sliderHolder = document.querySelector(".slider-holder");

const next = document.querySelector(".right");
const prev = document.querySelector(".left");

document.addEventListener("DOMContentLoaded", function () {
  fetchQuestions();
});

submitButton.style.display = "none";
sliderHolder.style.display = "block";

function fetchQuestions() {
  fetch("/questions")
    .then((response) => response.json())
    .then((data) => {
      console.log(data,'questions')
      createQuestions(data);
      questions = data;
      slide(curSlide);
    })
    .then(() => {
      //   showSlides(slideIndex);
    })
    .catch((error) => console.error("Error fetching questions:", error));
}

function createSlides(questionData) {
  if (!questionData) return; // Check if questionData is null or undefined

  const wrapper = document.createElement("div");
  wrapper.classList.add("slide-wrapper");
  const slide = document.createElement("div");
  slide.classList.add("slide");
  const colorContainer = document.createElement("div");
  colorContainer.classList.add("color-container");

  // Check if questionData.questionID, questionData.question, questionData.optionA, etc. are not null or undefined
  if (!questionData.questionID || !questionData.question || !questionData.optionA || !questionData.optionB || !questionData.optionC || !questionData.optionD) {
    console.error('Invalid question data');
    return;
  }

  // Creating HTML for question and options
  const card = `<div>
    <h3>${questionData.question}</h3>
    <div>
      <label>
        <input type="radio" name="${questionData.questionID}" value="${questionData.optionA}" onclick="saveAnswer('${questionData.question.replace(/'/g, "\\'")}', '${questionData.optionA.replace(/'/g, "\\'")}')">
        ${questionData.optionA}
      </label>
    </div>
    <div>
      <label>
        <input type="radio" name="${questionData.questionID}" value="${questionData.optionB}" onclick="saveAnswer('${questionData.question.replace(/'/g, "\\'")}', '${questionData.optionB.replace(/'/g, "\\'")}')">
        ${questionData.optionB}
      </label>
    </div>
    <div>
      <label>
        <input type="radio" name="${questionData.questionID}" value="${questionData.optionC}" onclick="saveAnswer('${questionData.question.replace(/'/g, "\\'")}', '${questionData.optionC.replace(/'/g, "\\'")}')">
        ${questionData.optionC}
      </label>
    </div>
    <div>
      <label>
        <input type="radio" name="${questionData.questionID}" value="${questionData.optionD}" onclick="saveAnswer('${questionData.question.replace(/'/g, "\\'")}', '${questionData.optionD.replace(/'/g, "\\'")}')">
        ${questionData.optionD}
      </label>
    </div>
</div>`;

  // Setting the HTML content to the colorContainer
  colorContainer.innerHTML = card;

  // Appending elements to the DOM
  slide.appendChild(colorContainer);
  wrapper.appendChild(slide);
  container.appendChild(wrapper);
}



function createQuestions(data) {
  data.forEach((question) => {
    createSlides(question);
  });
}

let curSlide = 0;

function slide(pos) {
  container.style.left = `-${pos * 100}vw`;
  if (curSlide === questions.length - 1) {
    next.style.display = "none";
    submitButton.style.display = "block";
  } else {
    next.style.display = "block";
  }
  if (curSlide === 0) {
    prev.style.display = "none";
  } else {
    prev.style.display = "block";
  }
}

function nextSlide() {
  const slides = document.querySelectorAll(".slide-wrapper");
  if (curSlide < slides.length - 1) {
    curSlide++;
    slide(curSlide);
  }
}

function prevSlide() {
  if (curSlide > 0) {
    curSlide--;
    slide(curSlide);
  }
}

next.addEventListener("click", nextSlide);

prev.addEventListener("click", (e) => {
  prevSlide();
});

function saveAnswer(que, ans) {
  answers[curSlide] = { question: que, answer: ans };
}

function sendDataToBackend(url, data) {
  console.log(url, data);
  fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  })
    .then((response) => {
      if (response.ok) {
        return response.json();
      }
      throw new Error("Network response was not ok.");
    })
    .then((data) => {
      console.log("Data sent successfully:", data);
        // You can add further logic here if needed
        window.location.href = '/chat'
    })
    .catch((error) => {
      console.error("There was a problem sending the data:", error);
    });
}

submitButton.addEventListener("click", () => {
  const exactLength = answers.filter(Boolean).length;

  console.log(answers.length, exactLength);

  if (answers.length == exactLength && answers.length != 0) {
    console.log("post Request");
    sendDataToBackend("/updateProfile", answers);
  } else {
    alert("Please Attend All Question");
  }
});
