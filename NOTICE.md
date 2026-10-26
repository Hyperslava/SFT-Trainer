# Сторонние данные и атрибуция

## CodeLLMExp

- **Что это**: датасет для локализации и объяснения уязвимостей в
  AI-сгенерированном коде (Python, Java, C), 10 403 записи.
- **Лицензия**: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/
- **Источник**: укажите ссылку на оригинальный репозиторий/публикацию авторов
  датасета (CodeLLMExp: A Multi-language Dataset for Automated Vulnerability
  Localization and Explanation in AI-Generated Code).
- **Изменения при распространении в этом репозитории**:
  - датасет упакован в `assets/codellmexp.zip`;
  - удалены служебные файлы Windows (`*:Zone.Identifier`) и `.DS_Store`;
  - добавлена контрольная сумма `assets/SHA256SUMS`;
  - файлы не изменялись по содержимому.

При использовании датасета в публикациях цитируйте работу авторов CodeLLMExp.

## Модели

Веса базовых моделей (Qwen, Llama и т. п.) в репозиторий не входят и
скачиваются пользователем с HuggingFace Hub на условиях их собственных лицензий.
