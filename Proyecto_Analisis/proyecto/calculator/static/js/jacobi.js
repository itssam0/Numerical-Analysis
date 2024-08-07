function generarMatrizA() {
    const filas = document.getElementById("filas").value;
    const tabla = document.getElementById("matrizInput");
    tabla.innerHTML = "";

    for (let i = 0; i < filas; i++) {
        let fila = document.createElement("tr");
        for (let j = 0; j < filas; j++) {
            let celda = document.createElement("td");
            let input = document.createElement("input");
            input.type = "number";
            input.step = "any";
            input.name = `A${i}${j}`;
            input.className = "form-control";
            celda.appendChild(input);
            fila.appendChild(celda);
        }
        tabla.appendChild(fila);
    }
}

function generarMatrizb() {
    const filas = document.getElementById("filas").value;
    const tabla = document.getElementById("matrizInputB");
    tabla.innerHTML = "";

    for (let i = 0; i < filas; i++) {
        let fila = document.createElement("tr");
        let celda = document.createElement("td");
        let input = document.createElement("input");
        input.type = "number";
        input.step = "any";
        input.name = `b${i}`;
        input.className = "form-control";
        celda.appendChild(input);
        fila.appendChild(celda);
        tabla.appendChild(fila);
    }
}

function generarMatrizx0() {
    const filas = document.getElementById("filas").value;
    const tabla = document.getElementById("matrizInputx0");
    tabla.innerHTML = "";

    for (let i = 0; i < filas; i++) {
        let fila = document.createElement("tr");
        let celda = document.createElement("td");
        let input = document.createElement("input");
        input.type = "number";
        input.step = "any";
        input.name = `x0${i}`;
        input.className = "form-control";
        celda.appendChild(input);
        fila.appendChild(celda);
        tabla.appendChild(fila);
    }
}

function guardarMatrices() {
    const filas = document.getElementById("filas").value;

    let matrizA = [];
    for (let i = 0; i < filas; i++) {
        let fila = [];
        for (let j = 0; j < filas; j++) {
            let valor = document.querySelector(`input[name='A${i}${j}']`).value || 0;
            fila.push(parseFloat(valor));
        }
        matrizA.push(fila);
    }
    document.getElementById("matrizA").value = JSON.stringify(matrizA);

    let vectorb = [];
    for (let i = 0; i < filas; i++) {
        let valor = document.querySelector(`input[name='b${i}']`).value || 0;
        vectorb.push(parseFloat(valor));
    }
    document.getElementById("vectorb").value = JSON.stringify(vectorb);

    let vectorx0 = [];
    for (let i = 0; i < filas; i++) {
        let valor = document.querySelector(`input[name='x0${i}']`).value || 0;
        vectorx0.push(parseFloat(valor));
    }
    document.getElementById("vectorx0").value = JSON.stringify(vectorx0);

    document.getElementById("filas-hidden").value = filas;
    document.getElementById("formMatrices").submit(); // Enviar el formulario después de guardar los datos
}
