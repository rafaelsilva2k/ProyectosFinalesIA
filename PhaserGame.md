#JUEGO Phaser

[Video](https://youtu.be/TgGdcbs2UqA?si=J9jgHrj_JmRs3w03)

##Codigo:

´´´
var w = 800;
var h = 400;
var jugador;
var fondo;

var bala, balaD = false, nave;
var bala2, bala2D = false, nave2;

var estatusRegreso = false;
var regresando = false;
var esquivando = false;

var esquivar;
var salto;
var menu;


var velocidadBala2;
var velocidadBala;
var despBala;
var despBala2;
var estatusAire;
var estatuSuelo;

// Variables para el modelo
var nnNetwork, nnEntrenamiento, nnSalida, datosEntrenamiento = [];
var modoAuto = false, eCompleto = false;

var juego = new Phaser.Game(w, h, Phaser.CANVAS, '', { preload: preload, create: create, update: update, render: render });

function preload() {
    juego.load.image('fondo', 'assets/game/fondo.jpg');
    juego.load.spritesheet('mono', 'assets/sprites/altair.png', 32, 48);
    juego.load.image('nave', 'assets/game/ufo.png');
    juego.load.image('bala', 'assets/sprites/purple_ball.png');
    juego.load.image('nave2', 'assets/game/ufo.png');
    juego.load.image('bala2', 'assets/sprites/purple_ball.png');
    juego.load.image('menu', 'assets/game/menu.png');
}

function create() {

    juego.physics.startSystem(Phaser.Physics.ARCADE);
    juego.physics.arcade.gravity.y = 800;
    juego.time.desiredFps = 30;

    fondo = juego.add.tileSprite(0, 0, w, h, 'fondo');
    nave = juego.add.sprite(w - 100, h - 70, 'nave');
    bala = juego.add.sprite(w - 100, h, 'bala');
    nave2 = juego.add.sprite(w - 766, h - 400, 'nave2');
    bala2 = juego.add.sprite(w - 725, h - 350, 'bala2');
    jugador = juego.add.sprite(50, h, 'mono');


    juego.physics.enable(jugador);
    jugador.body.collideWorldBounds = true;
    var corre = jugador.animations.add('corre', [8, 9, 10, 11]);
    jugador.animations.play('corre', 10, true);

    juego.physics.enable(bala);
    juego.physics.enable(bala2);
    bala.body.collideWorldBounds = true;
    bala2.body.collideWorldBounds = true;

    pausaL = juego.add.text(w - 100, 20, 'Pausa', { font: '20px Arial', fill: '#fff' });
    pausaL.inputEnabled = true;
    pausaL.events.onInputUp.add(pausa, self);
    juego.input.onDown.add(mPausa, self);

    salto = juego.input.keyboard.addKey(Phaser.Keyboard.SPACEBAR);
    esquivar = juego.input.keyboard.addKey(Phaser.Keyboard.LEFT);


    nnNetwork = new synaptic.Architect.Perceptron(4, 8, 4, 2);
    nnEntrenamiento = new synaptic.Trainer(nnNetwork);

}

function enRedNeural() {
    nnEntrenamiento.train(datosEntrenamiento, { rate: 0.0006, iterations: 70000, shuffle: true });
}


function datosDeEntrenamiento(param_entrada) {
    console.log("Entrada", param_entrada[0] + " " + param_entrada[1], param_entrada[2] + " " + param_entrada[3]);
    nnSalida = nnNetwork.activate(param_entrada);
    salida = [nnSalida[0] * 100, nnSalida[1] * 100]
    return salida;
}

function pausa() {
    juego.paused = true;
    menu = juego.add.sprite(w / 2, h / 2, 'menu');
    menu.anchor.setTo(0.5, 0.5);
}

function mPausa(event) {
    if (juego.paused) {
        var menu_x1 = w / 2 - 270 / 2, menu_x2 = w / 2 + 270 / 2,
            menu_y1 = h / 2 - 180 / 2, menu_y2 = h / 2 + 180 / 2;

        var mouse_x = event.x,
            mouse_y = event.y;

        if (mouse_x > menu_x1 && mouse_x < menu_x2 && mouse_y > menu_y1 && mouse_y < menu_y2) {
            if (mouse_x >= menu_x1 && mouse_x <= menu_x2 && mouse_y >= menu_y1 && mouse_y <= menu_y1 + 90) {
                eCompleto = false;
                datosEntrenamiento = [];
                modoAuto = false;
            } else if (mouse_x >= menu_x1 && mouse_x <= menu_x2 && mouse_y >= menu_y1 + 90 && mouse_y <= menu_y2) {
                if (!eCompleto) {
                    console.log("", "Entrenamiento " + datosEntrenamiento.length + " valores");
                    enRedNeural();
                    eCompleto = true;
                }
                modoAuto = true;
            }

            menu.destroy();
            resetJugador();
            resetVariables();
            resetY();
            juego.paused = false;

        }
    }
}

function disparo() {
    velocidadBala = -1 * velocidadRandom(300, 800);
    bala.body.velocity.y = 0;
    bala.body.velocity.x = velocidadBala;
    balaD = true;
}

function disparo2() {
    bala2.body.velocity.x = 0;
    bala2D = true;
}

function resetVariables() {
    bala.body.velocity.x = 0;
    bala.position.x = w - 100;
    balaD = false;
}

function resetJugador() {
    jugador.position.x = 50;
    jugador.position.y = h - 50;
    esquivando = false;
}

function resetY() {
    bala2.position.y = h - 350;
    bala2.position.x = w - 725;
    bala2.body.velocity.y = 0;
    bala2.body.velocity.x = 0;
    bala2D = false;
}

function update() {
    fondo.tilePosition.x -= 1;

    juego.physics.arcade.collide([bala, bala2], jugador, colisionH, null, this);

    velocidadBala2 = Math.round(bala2.body.velocity.y);
    despBala = Math.floor(jugador.position.x - bala.position.x);
    despBala2 = Math.floor(jugador.position.y - bala2.position.y);

    if (modoAuto == false) {
        estatusAire = 0;
        estatusMover = 0;

        if (!jugador.body.onFloor()) {
            estatusAire = 1;
        }

        if (estatusRegreso) {
            estatusMover = 1;
        }
        if (salto.isDown && jugador.body.onFloor()) {
            jugador.body.velocity.y = -270;
        }
        if (esquivar.isDown && !esquivando) {
            jugador.body.velocity.x = -150;
            estatusRegreso = true;
            esquivando = true;
        }
       
        if (jugador.position.x == 0 && esquivando) {
            jugador.body.velocity.x = 150;
            estatusRegreso = false;
            regresando = true;
        }
        if (jugador.position.x == 50 && regresando) {
            jugador.position.x = 50;
            jugador.body.velocity.x = 0;
            regresando = false;
            esquivando = false;
        }
    } else {
        movimientos = datosDeEntrenamiento([despBala, velocidadBala, despBala2, velocidadBala2])
        console.log("movimientos: ", movimientos)
       
        if (movimientos[0] > 60 && jugador.body.onFloor()) { 
            jugador.body.velocity.y = -270;
        }
        if (movimientos[1] > 60 && !esquivando) {
            jugador.body.velocity.x = -150;
            esquivando = true;
        }
        
        if (jugador.position.x == 0 && esquivando) {
            jugador.body.velocity.x = 150;
            regresando = true;
        }
        if (jugador.position.x == 50 && regresando) {
            jugador.position.x = 50;
            jugador.body.velocity.x = 0;
            regresando = false;
            esquivando = false;
        }
    }

    if (balaD == false) {
        disparo();
    }

    if (bala2D == false) {
        disparo2();
    }

    if (bala2.position.y >= 370) {
        resetY();
    }

    if (bala.position.x <= 0) {
        resetVariables();
    }

    if (modoAuto == false && (bala.position.x > 0 || bala2.position.y < 371)) {

        datosEntrenamiento.push({
            'input': [despBala, velocidadBala, despBala2, velocidadBala2],
            'output': [estatusAire, estatusMover]
        });

        console.log(despBala, + "\t" + velocidadBala, + "\t" + estatusAire, + "\t" + despBala2, + "\t" + velocidadBala2, + "\t" + estatusMover);
    }

}

function colisionH() {
    pausa();
}

function velocidadRandom(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

function render() {

}
