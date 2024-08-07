function generarMatrizA() {
    var filas = parseInt(document.getElementById("filas").value);
    var tabla = document.getElementById("matrizInput");

    tabla.innerHTML = '';

    for (var i = 0; i < filas; i++) {
        var fila = tabla.insertRow(i);

        for (var j = 0; j < filas; j++) {
            var celda = fila.insertCell(j);
            var input = document.createElement("input");
            input.type = "number";
            input.name = "A" + i + j;
            input.className = "form-control";
            celda.appendChild(input);
        }
    }
}

function generarMatrizb() {
    var filas = parseInt(document.getElementById("filas").value);
    var tabla = document.getElementById("matrizInputB");

    tabla.innerHTML = '';

    for (var i = 0; i < filas; i++) {
        var fila = tabla.insertRow(i);
        var celda = fila.insertCell(0);
        var input = document.createElement("input");
        input.type = "number";
        input.name = "b" + i;
        input.className = "form-control";
        celda.appendChild(input);
    }
}

function generarMatrizx0() {
    var filas = parseInt(document.getElementById("filas").value);
    var tabla = document.getElementById("matrizInputx0");

    tabla.innerHTML = '';

    for (var i = 0; i < filas; i++) {
        var fila = tabla.insertRow(i);
        var celda = fila.insertCell(0);
        var input = document.createElement("input");
        input.type = "number";
        input.name = "x0" + i;
        input.className = "form-control";
        celda.appendChild(input);
    }
}

function guardarMatrices() {
    var filas = parseInt(document.getElementById("filas").value);

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
    document.getElementById("matrizB").value = JSON.stringify(vectorb);

    let vectorx0 = [];
    for (let i = 0; i < filas; i++) {
        let valor = document.querySelector(`input[name='x0${i}']`).value || 0;
        vectorx0.push(parseFloat(valor));
    }
    document.getElementById("matrizX0").value = JSON.stringify(vectorx0);

    document.getElementById("filas-hidden").value = filas;

    document.getElementById("formMatrices").submit();
}
