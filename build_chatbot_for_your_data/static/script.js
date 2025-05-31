//let lightMode = true;
//let isFirstMessage = true;
//const baseUrl = window.location.origin;
//
//// Show/hide loading animations
//async function showBotLoadingAnimation() {
//  await sleep(200);
//  $(".loading-animation")[1].style.display = "inline-block";
//  document.getElementById('send-button').disabled = true;
//}
//
//function hideBotLoadingAnimation() {
//  $(".loading-animation")[1].style.display = "none";
//  if(!isFirstMessage) {
//    document.getElementById('send-button').disabled = false;
//  }
//}
//
//async function showUserLoadingAnimation() {
//  await sleep(100);
//  $(".loading-animation")[0].style.display = "flex";
//}
//
//function hideUserLoadingAnimation() {
//  $(".loading-animation")[0].style.display = "none";
//}
//
//// Process user message with server
//const processUserMessage = async (userMessage) => {
//  let response = await fetch(baseUrl + "/process-message", {
//    method: "POST",
//    headers: {
//      "Accept": "application/json",
//      "Content-Type": "application/json"
//    },
//    body: JSON.stringify({ userMessage: userMessage }),
//  });
//
//  if (!response.ok) {
//    const error = await response.json();
//    throw new Error(error.botResponse || "Server error");
//  }
//
//  return await response.json();
//};
//
//// Utility functions
//const cleanTextInput = (value) => {
//  return value
//    .trim()
//    .replace(/[\n\t]/g, "")
//    .replace(/<[^>]*>/g, "")
//    .replace(/[<>&;]/g, "");
//};
//
//const sleep = (time) => new Promise((resolve) => setTimeout(resolve, time));
//
//const scrollToBottom = () => {
//  $("#chat-window").animate({
//    scrollTop: $("#chat-window")[0].scrollHeight,
//  });
//};
//
//// Display user message in chat
//const populateUserMessage = (userMessage) => {
//  $("#message-input").val("");
//
//  $("#message-list").append(
//    `<div class='message-line my-text'>
//      <div class='message-box my-text${!lightMode ? " dark" : ""}'>
//        <div class='me'>${userMessage}</div>
//      </div>
//    </div>`
//  );
//
//  scrollToBottom();
//};
//
//// Display bot response in chat
//const populateBotResponse = async (userMessage) => {
//  await showBotLoadingAnimation();
//
//  let response;
//  let uploadButtonHtml = '';
//
//  if (isFirstMessage) {
//    response = {
//      botResponse: "Hello there! I'm your friendly data assistant. Please upload a PDF file for me to analyze."
//    };
//    uploadButtonHtml = `
//      <input type="file" id="file-upload" accept=".pdf" hidden>
//      <button id="upload-button" class="btn btn-primary btn-sm">Upload File</button>
//    `;
//  } else {
//    try {
//      response = await processUserMessage(userMessage);
//    } catch (error) {
//      response = { botResponse: `Error: ${error.message}` };
//    }
//  }
//
//  renderBotResponse(response, uploadButtonHtml);
//
//  // Setup file upload handling
//  if (isFirstMessage) {
//    $("#upload-button").on("click", function() {
//      $("#file-upload").click();
//    });
//
//    $("#file-upload").on("change", async function() {
//      const file = this.files[0];
//      await showBotLoadingAnimation();
//
//      const formData = new FormData();
//      formData.append('file', file);
//
//      try {
//        let response = await fetch(baseUrl + "/process-document", {
//          method: "POST",
//          body: formData,
//        });
//
//        if (response.ok) {
//          document.querySelector('#upload-button').disabled = true;
//          response = await response.json();
//          renderBotResponse(response, '');
//        } else {
//          const error = await response.json();
//          throw new Error(error.botResponse || "Upload failed");
//        }
//      } catch (error) {
//        $("#message-list").append(
//          `<div class='message-line'>
//            <div class='message-box${!lightMode ? " dark" : ""}'>
//              Error: ${error.message}
//            </div>
//          </div>`
//        );
//        scrollToBottom();
//      } finally {
//        hideBotLoadingAnimation();
//      }
//    });
//
//    isFirstMessage = false;
//  }
//};
//
//// Render bot response in chat
//const renderBotResponse = (response, uploadButtonHtml) => {
//  hideBotLoadingAnimation();
//
//  $("#message-list").append(
//    `<div class='message-line'>
//      <div class='message-box${!lightMode ? " dark" : ""}'>
//        ${response.botResponse.trim()}<br>${uploadButtonHtml}
//      </div>
//    </div>`
//  );
//
//  scrollToBottom();
//}
//
//// Initialize chat
//populateBotResponse();
//
//// DOM Ready Handler
//$(document).ready(function() {
//  document.getElementById('send-button').disabled = true;
//
//  // Enter key handling
//  $("#message-input").keyup(function(event) {
//    let inputVal = cleanTextInput($("#message-input").val());
//
//    if (event.keyCode === 13 && inputVal !== "") {
//      populateUserMessage(inputVal);
//      populateBotResponse(inputVal);
//    }
//  });
//
//  // Send button handling
//  $("#send-button").click(async function() {
//    const message = cleanTextInput($("#message-input").val());
//
//    if (message) {
//      populateUserMessage(message);
//      populateBotResponse(message);
//    }
//  });
//
//  // Reset chat handling
//  $("#reset-button").click(async function() {
//    $("#message-list").empty();
//    isFirstMessage = true;
//
//    if (document.querySelector('#upload-button')) {
//      document.querySelector('#upload-button').disabled = false;
//    }
//
//    populateBotResponse();
//  });
//
//  // Light/dark mode toggle
//  $("#light-dark-mode-switch").change(function() {
//    $("body").toggleClass("dark-mode");
//    $(".message-box").toggleClass("dark");
//    $(".loading-dots").toggleClass("dark");
//    $(".dot").toggleClass("dark-dot");
//    lightMode = !lightMode;
//  });
//});

let lightMode = true;
let isFirstMessage = true;
const baseUrl = window.location.origin;

// Initialize GSAP
gsap.registerPlugin(ScrollTrigger);

// Show/hide loading animations
async function showBotLoadingAnimation() {
    await sleep(200);
    gsap.to(".typing-indicator", {
        duration: 0.3,
        opacity: 1,
        y: 0,
        display: "flex",
        ease: "power2.out"
    });
    document.getElementById('send-button').disabled = true;
}

function hideBotLoadingAnimation() {
    gsap.to(".typing-indicator", {
        duration: 0.2,
        opacity: 0,
        y: 10,
        display: "none",
        ease: "power2.in",
        onComplete: () => {
            if(!isFirstMessage) {
                document.getElementById('send-button').disabled = false;
            }
        }
    });
}

// Process user message with server
const processUserMessage = async (userMessage) => {
    let response = await fetch(baseUrl + "/process-message", {
        method: "POST",
        headers: {
            "Accept": "application/json",
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ userMessage: userMessage }),
    });

    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.botResponse || "Server error");
    }

    return await response.json();
};

// Utility functions
const cleanTextInput = (value) => {
    return value
        .trim()
        .replace(/[\n\t]/g, "")
        .replace(/<[^>]*>/g, "")
        .replace(/[<>&;]/g, "");
};

const sleep = (time) => new Promise((resolve) => setTimeout(resolve, time));

const scrollToBottom = () => {
    const chatWindow = document.getElementById('chat-window');
    gsap.to(chatWindow, {
        duration: 0.5,
        scrollTo: {y: chatWindow.scrollHeight, autoKill: false},
        ease: "power2.out"
    });
};

// Display user message in chat
const populateUserMessage = (userMessage) => {
    $("#message-input").val("");

    const messageElement = $(`
        <div class='message-line user'>
            <div class='message-box user'>
                <div class='message-content'>${userMessage}</div>
            </div>
        </div>
    `);

    $("#message-list").append(messageElement);

    // Animate message entry
    gsap.fromTo(messageElement,
        { opacity: 0, y: 20 },
        {
            opacity: 1,
            y: 0,
            duration: 0.4,
            ease: "back.out(1.2)",
            onComplete: scrollToBottom
        }
    );
};

// Display bot response in chat
const populateBotResponse = async (userMessage) => {
    await showBotLoadingAnimation();

    let response;
    let uploadButtonHtml = '';

    if (isFirstMessage) {
        response = {
            botResponse: "Hello there! I'm your friendly data assistant. Please upload a PDF file for me to analyze."
        };
        uploadButtonHtml = `
            <input type="file" id="file-upload" accept=".pdf" hidden>
            <button id="upload-button" class="btn animate__animated animate__pulse animate__infinite mt-3">
                <i class="fas fa-cloud-upload-alt mr-2"></i>Upload PDF
            </button>
        `;
    } else {
        try {
            response = await processUserMessage(userMessage);
        } catch (error) {
            response = { botResponse: `Error: ${error.message}` };
        }
    }

    renderBotResponse(response, uploadButtonHtml);

    // Setup file upload handling
    if (isFirstMessage) {
        $("#upload-button").on("click", function() {
            $("#file-upload").click();
        });

        $("#file-upload").on("change", async function() {
            const file = this.files[0];
            await showBotLoadingAnimation();

            const formData = new FormData();
            formData.append('file', file);

            try {
                let response = await fetch(baseUrl + "/process-document", {
                    method: "POST",
                    body: formData,
                });

                if (response.ok) {
                    document.querySelector('#upload-button').disabled = true;
                    response = await response.json();
                    renderBotResponse(response, '');
                } else {
                    const error = await response.json();
                    throw new Error(error.botResponse || "Upload failed");
                }
            } catch (error) {
                const errorElement = $(`
                    <div class='message-line bot'>
                        <div class='message-box bot'>
                            Error: ${error.message}
                        </div>
                    </div>
                `);

                $("#message-list").append(errorElement);
                animateMessageEntry(errorElement);
            } finally {
                hideBotLoadingAnimation();
            }
        });

        isFirstMessage = false;
    }
};

// Animate message entry
const animateMessageEntry = (element) => {
    gsap.fromTo(element,
        { opacity: 0, y: 20 },
        {
            opacity: 1,
            y: 0,
            duration: 0.4,
            ease: "back.out(1.2)",
            onComplete: scrollToBottom
        }
    );
};

// Render bot response in chat
const renderBotResponse = (response, uploadButtonHtml) => {
    hideBotLoadingAnimation();

    const messageElement = $(`
        <div class='message-line bot'>
            <div class='message-box bot'>
                <div class='message-content'>${response.botResponse.trim()}</div>
                ${uploadButtonHtml}
            </div>
        </div>
    `);

    $("#message-list").append(messageElement);
    animateMessageEntry(messageElement);
}

// Initialize chat
$(document).ready(function() {
    // Initialize with animation
    gsap.from("h1", {duration: 1, y: -50, opacity: 0, ease: "bounce.out"});
    gsap.from("#chat-window", {duration: 0.8, y: 30, opacity: 0, delay: 0.3});
    gsap.from(".input-group", {duration: 0.8, y: 30, opacity: 0, delay: 0.5});

    document.getElementById('send-button').disabled = true;

    // Enter key handling
    $("#message-input").keyup(function(event) {
        let inputVal = cleanTextInput($("#message-input").val());
        document.getElementById('send-button').disabled = !inputVal;

        if (event.keyCode === 13 && inputVal !== "") {
            populateUserMessage(inputVal);
            populateBotResponse(inputVal);
        }
    });

    // Input field enable/disable button
    $("#message-input").on('input', function() {
        let inputVal = cleanTextInput($(this).val());
        document.getElementById('send-button').disabled = !inputVal;
    });

    // Send button handling
    $("#send-button").click(async function() {
        const message = cleanTextInput($("#message-input").val());
        if (message) {
            // Button animation
            gsap.to(this, {
                duration: 0.2,
                scale: 0.9,
                yoyo: true,
                repeat: 1,
                ease: "power2.inOut"
            });

            populateUserMessage(message);
            populateBotResponse(message);
        }
    });

    // Reset chat handling
    $("#reset-button").click(async function() {
        // Button animation
        gsap.to(this, {
            duration: 0.3,
            rotation: 360,
            ease: "power2.out"
        });

        // Animate chat clearing
        gsap.to("#message-list > *", {
            duration: 0.3,
            opacity: 0,
            y: -20,
            stagger: 0.05,
            onComplete: () => {
                $("#message-list").empty();
                isFirstMessage = true;
                if (document.querySelector('#upload-button')) {
                    document.querySelector('#upload-button').disabled = false;
                }
                populateBotResponse();
            }
        });
    });

    // Light/dark mode toggle
    $("#light-dark-mode-switch").change(function() {
        $("body").toggleClass("dark-mode");

        // Animate transition
        gsap.to("body", {duration: 0.5, backgroundColor: $("body").css("background-color")});

        // Toggle switch animation
        const switchBtn = $(this);
        if (switchBtn.prop('checked')) {
            gsap.to('.custom-control-label .light-icon', {duration: 0.3, opacity: 0, onComplete: () => {
                gsap.to('.custom-control-label .dark-icon', {duration: 0.3, opacity: 1});
            }});
        } else {
            gsap.to('.custom-control-label .dark-icon', {duration: 0.3, opacity: 0, onComplete: () => {
                gsap.to('.custom-control-label .light-icon', {duration: 0.3, opacity: 1});
            }});
        }
    });

    // Initial bot message
    setTimeout(() => {
        populateBotResponse();
    }, 1000);
});