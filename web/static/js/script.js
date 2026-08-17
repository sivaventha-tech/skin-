/* ============================================================
   SKIN DISEASE AI
   IMAGE UPLOAD + PREVIEW + REMOVE + DRAG & DROP
   ============================================================ */

document.addEventListener("DOMContentLoaded", function () {

    /* ========================================================
       ELEMENTS
    ======================================================== */

    const form = document.getElementById("predictionForm");

    const imageInput = document.getElementById("imageInput");

    const browseButton = document.getElementById("browseButton");

    const dropZone = document.getElementById("dropZone");

    const previewContainer =
        document.getElementById("previewContainer");

    const imagePreview =
        document.getElementById("imagePreview");

    const fileName =
        document.getElementById("fileName");

    const fileSize =
        document.getElementById("fileSize");

    const removeImageButton =
        document.getElementById("removeImageButton");

    const analyzeButton =
        document.getElementById("analyzeButton");

    const loadingContainer =
        document.getElementById("loadingContainer");

    const buttonText =
        document.getElementById("buttonText");

    const buttonIcon =
        document.getElementById("buttonIcon");

    const uploadTitle =
        document.getElementById("uploadTitle");

    const uploadSubtitle =
        document.getElementById("uploadSubtitle");

    const uploadIcon =
        document.getElementById("uploadIcon");


    /* ========================================================
       CONFIGURATION
    ======================================================== */

    const MAX_FILE_SIZE = 10 * 1024 * 1024;

    const ALLOWED_TYPES = [
        "image/jpeg",
        "image/png",
        "image/webp"
    ];


    /* ========================================================
       CHECK ELEMENTS
    ======================================================== */

    if (!imageInput) {
        console.error("imageInput not found.");
        return;
    }

    if (!browseButton) {
        console.error("browseButton not found.");
        return;
    }

    if (!removeImageButton) {
        console.error("removeImageButton not found.");
        return;
    }


    /* ========================================================
       BROWSE BUTTON
    ======================================================== */

    browseButton.addEventListener(
        "click",
        function (event) {

            event.preventDefault();

            imageInput.click();

        }
    );


    /* ========================================================
       FILE INPUT CHANGE
    ======================================================== */

    imageInput.addEventListener(
        "change",
        function () {

            if (
                imageInput.files &&
                imageInput.files.length > 0
            ) {

                const file =
                    imageInput.files[0];

                handleSelectedFile(file);

            }

        }
    );


    /* ========================================================
       HANDLE SELECTED FILE
    ======================================================== */

    function handleSelectedFile(file) {

        /* ----------------------------------------------------
           Check file type
        ---------------------------------------------------- */

        if (
            !ALLOWED_TYPES.includes(
                file.type
            )
        ) {

            alert(
                "Invalid image format.\n\n" +
                "Please select JPG, JPEG, PNG or WEBP."
            );

            clearImage();

            return;
        }


        /* ----------------------------------------------------
           Check file size
        ---------------------------------------------------- */

        if (
            file.size > MAX_FILE_SIZE
        ) {

            alert(
                "Image is too large.\n\n" +
                "Maximum allowed size is 10 MB."
            );

            clearImage();

            return;
        }


        /* ----------------------------------------------------
           Create image preview
        ---------------------------------------------------- */

        const reader =
            new FileReader();


        reader.onload = function (event) {

            imagePreview.src =
                event.target.result;


            /* Show preview */

            previewContainer.style.display =
                "block";


            /* Enable analyze button */

            analyzeButton.disabled =
                false;


            /* Update filename */

            fileName.textContent =
                file.name;


            /* Update file size */

            fileSize.textContent =
                formatFileSize(
                    file.size
                );


            /* Update upload area */

            uploadTitle.textContent =
                "Image selected successfully";

            uploadSubtitle.textContent =
                "You can remove it and choose another image.";

            uploadIcon.textContent =
                "✓";


            /* Scroll slightly to preview */

            setTimeout(
                function () {

                    previewContainer.scrollIntoView({
                        behavior: "smooth",
                        block: "nearest"
                    });

                },
                100
            );

        };


        reader.onerror = function () {

            alert(
                "Unable to read the selected image."
            );

            clearImage();

        };


        reader.readAsDataURL(file);
    }


    /* ========================================================
       REMOVE IMAGE
    ======================================================== */

    removeImageButton.addEventListener(
        "click",
        function (event) {

            event.preventDefault();

            event.stopPropagation();

            clearImage();

        }
    );


    /* ========================================================
       CLEAR IMAGE
    ======================================================== */

    function clearImage() {

        /* ----------------------------------------------------
           IMPORTANT:
           Reset the actual file input.
           Without this, the browser still keeps the file.
        ---------------------------------------------------- */

        imageInput.value = "";


        /* ----------------------------------------------------
           Remove image preview
        ---------------------------------------------------- */

        imagePreview.src = "";


        /* ----------------------------------------------------
           Hide preview container
        ---------------------------------------------------- */

        previewContainer.style.display =
            "none";


        /* ----------------------------------------------------
           Disable analyze button
        ---------------------------------------------------- */

        analyzeButton.disabled =
            true;


        /* ----------------------------------------------------
           Clear file information
        ---------------------------------------------------- */

        fileName.textContent = "";

        fileSize.textContent = "";


        /* ----------------------------------------------------
           Restore upload area
        ---------------------------------------------------- */

        uploadTitle.textContent =
            "Drag & Drop your image here";

        uploadSubtitle.textContent =
            "or click below to browse files";

        uploadIcon.textContent =
            "☁️";


        /* ----------------------------------------------------
           Make sure drop zone is normal
        ---------------------------------------------------- */

        dropZone.classList.remove(
            "drag-over"
        );


        console.log(
            "Image removed successfully."
        );
    }


    /* ========================================================
       DRAG OVER
    ======================================================== */

    dropZone.addEventListener(
        "dragover",
        function (event) {

            event.preventDefault();

            event.stopPropagation();

            dropZone.classList.add(
                "drag-over"
            );

        }
    );


    /* ========================================================
       DRAG LEAVE
    ======================================================== */

    dropZone.addEventListener(
        "dragleave",
        function (event) {

            event.preventDefault();

            event.stopPropagation();

            dropZone.classList.remove(
                "drag-over"
            );

        }
    );


    /* ========================================================
       DROP
    ======================================================== */

    dropZone.addEventListener(
        "drop",
        function (event) {

            event.preventDefault();

            event.stopPropagation();

            dropZone.classList.remove(
                "drag-over"
            );


            const files =
                event.dataTransfer.files;


            if (
                files &&
                files.length > 0
            ) {

                const file =
                    files[0];


                /* --------------------------------------------
                   Put dropped file into input
                   -------------------------------------------- */

                try {

                    const dataTransfer =
                        new DataTransfer();

                    dataTransfer.items.add(
                        file
                    );

                    imageInput.files =
                        dataTransfer.files;

                } catch (error) {

                    console.warn(
                        "Could not assign dropped file.",
                        error
                    );

                }


                handleSelectedFile(file);
            }

        }
    );


    /* ========================================================
       CLICK DROP ZONE
    ======================================================== */

    dropZone.addEventListener(
        "click",
        function (event) {

            /*
             * Don't trigger another click when the
             * Browse button itself was clicked.
             */

            if (
                event.target === browseButton
            ) {

                return;
            }


            /*
             * Don't trigger when clicking
             * inside an existing button.
             */

            if (
                event.target.closest(
                    "button"
                )
            ) {

                return;
            }


            imageInput.click();

        }
    );


    /* ========================================================
       FORM SUBMIT
    ======================================================== */

    if (form) {

        form.addEventListener(
            "submit",
            function (event) {

                /* --------------------------------------------
                   Make sure image exists
                -------------------------------------------- */

                if (
                    !imageInput.files ||
                    imageInput.files.length === 0
                ) {

                    event.preventDefault();

                    alert(
                        "Please select a skin image first."
                    );

                    return;
                }


                /* --------------------------------------------
                   Show loading state
                -------------------------------------------- */

                analyzeButton.disabled =
                    true;


                buttonIcon.textContent =
                    "⏳";


                buttonText.textContent =
                    "Analyzing Image...";


                loadingContainer.style.display =
                    "block";


                /* Prevent changing image during upload */

                browseButton.disabled =
                    true;

                removeImageButton.disabled =
                    true;

            }
        );

    }


    /* ========================================================
       FILE SIZE FORMATTER
    ======================================================== */

    function formatFileSize(bytes) {

        if (
            bytes === 0
        ) {

            return "0 Bytes";
        }


        const units = [
            "Bytes",
            "KB",
            "MB",
            "GB"
        ];


        const index =
            Math.floor(
                Math.log(bytes) /
                Math.log(1024)
            );


        const size =
            bytes /
            Math.pow(
                1024,
                index
            );


        return (
            size.toFixed(1)
            + " "
            + units[index]
        );

    }


    /* ========================================================
       INITIAL STATE
    ======================================================== */

    clearImage();


    console.log(
        "✓ Skin Disease AI JavaScript loaded."
    );

    console.log(
        "✓ Image upload enabled."
    );

    console.log(
        "✓ Image preview enabled."
    );

    console.log(
        "✓ Remove button enabled."
    );

    console.log(
        "✓ Drag & drop enabled."
    );

});