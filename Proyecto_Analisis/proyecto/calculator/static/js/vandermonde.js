function agregarCampos() {
    var cantidadPuntos = document.getElementById("cantidadPuntos").value;
    var camposPuntos = document.getElementById("camposPuntos");

    camposPuntos.innerHTML = '';  // Limpiar campos existentes

    for (var i = 0; i < cantidadPuntos; i++) {
        var divRow = document.createElement('div');
        divRow.className = 'row';

        var divColX = document.createElement('div');
        divColX.className = 'col-md-3';
        var inputX = document.createElement('input');
        inputX.type = 'text';
        inputX.className = 'form-control';
        inputX.name = 'x' + i;  // Cambiar el nombre para ser único
        inputX.placeholder = 'x' + (i + 1);
        divColX.appendChild(inputX);

        var divColY = document.createElement('div');
        divColY.className = 'col-md-4';
        var inputY = document.createElement('input');
        inputY.type = 'text';
        inputY.className = 'form-control';
        inputY.name = 'y' + i;  // Cambiar el nombre para ser único
        inputY.placeholder = 'y' + (i + 1);
        divColY.appendChild(inputY);

        divRow.appendChild(divColX);
        divRow.appendChild(divColY);
        camposPuntos.appendChild(divRow);
    }
}
