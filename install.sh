#!/bin/bash

check_virtualenv() {
    if ! command -v virtualenv &> /dev/null; then
        echo "virtualenv is not installed. Installing..."
        python3 -m pip install --user virtualenv
        echo "virtualenv installation complete."
    fi
}

export_deps() {
    local env_name=${1:-".ssdwsn_venv"}

    if [ ! -d "$env_name" ]; then
        echo "Virtual environment '$env_name' not found. Use '$0 create [env_name]' to create one."
        return 1
    fi

    source "./$env_name/bin/activate"
    pipreqs --force
    # pip freeze > requirements.txt
    echo "Dependencies exported to requirements.txt"
}

install() {
    # Check if virtualenv is installed, if not, install it
    pip install -U pip
    pip install pipreqs
    pip install Cython
    check_virtualenv
    
    local env_name=${1:-".ssdwsn_venv"}
    if [ -d "$env_name" ]; then
        echo "Virtual environment '$env_name' already exists. Aborting."
        source "./$env_name/bin/activate"

        if [ -f "requirements.txt" ]; then
            pip install -r ./requirements.txt
        fi

        if [ -f "setup.py" ]; then
            pip install -e .
        fi
        return 1
    fi

    python3 -m venv "$env_name"
    source "./$env_name/bin/activate"

    if [ -f "requirements.txt" ]; then
        pip install -r ./requirements.txt
    fi

    if [ -f "setup.py" ]; then
        pip install -e .
    fi
}

run() {
    local env_name=${1:-".ssdwsn_venv"}

    if [ ! -d "$env_name" ]; then
        echo "Virtual environment '$env_name' not found. Use '$0 install [env_name]' to install SSDWSN."
        return 1
    fi
    source "./$env_name/bin/activate"
    tensorboard --logdir output/logs &
    sudo python3 ssdwsn/main.py
}

clean() {

    local env_name=${1:-".ssdwsn_venv"}
    sudo rm -r build
    sudo rm -r dist
    sudo rm -r ssdwsn.egg-info
    sudo rm -r outputs/logs/*
}

uninstall() {
    local env_name=${1:-".ssdwsn_venv"}

    if [ ! -d "$env_name" ]; then
        echo "Virtual environment '$env_name' not found."
        return 1
    fi

    deactivate
    rm -rf "$env_name"
}

print_help() {
    echo "Usage: $0 [option] [env_name]"
    echo "Options:"
    echo "  install      Install SSDWSN"
    echo "  run      Run SSDWSN"
    echo "  clean      clean SSDWSN setup files and logs"
    echo "  uninstall    Uninstall SSDWSN"
    echo "  export      Export installed dependencies to requirements.txt within a virtual environment (default name: .ssdwsn_venv)"
}

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    print_help
    return 0
fi

case "$1" in
    "install")
        install "$2"
        ;;
    "run")
        run "$2"
        ;;
    "clean")
        clean "$2"
        ;;
    "uninstall")
        uninstall "$2"
        ;;
    "export")
        export_deps "$2"
        ;;
    *)
        echo "Unknown option: $1"
        print_help
        exit 1
        ;;
esac