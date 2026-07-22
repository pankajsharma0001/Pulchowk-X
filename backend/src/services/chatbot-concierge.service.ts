import locationData from "../data/campus_data.json" with { type: "json" };
import studentSupportKbData from "../data/student_support_kb.json" with {
  type: "json",
};
import { GoogleGenerativeAI } from "@google/generative-ai";
import ENV from "../config/ENV.js";
import {
  buildAppContext,
  type AppContextTopic,
} from "./chatbot-context.service.js";

export type ConciergeAction =
  | "show_route"
  | "show_location"
  | "show_multiple_locations"
  | "text_answer";

export type ConciergeIntent =
  | "route_navigation"
  | "location_lookup"
  | "process_howto"
  | "policy_query"
  | "service_lookup"
  | "office_lookup"
  | "deadline_query"
  | "escalation"
  | "unknown"
  | "notice_query"
  | "event_query"
  | "club_query"
  | "lost_found_query"
  | "marketplace_query"
  | "app_help"
  | "general_campus"
  | "greeting";

type SourceType = "official_page" | "office_confirmed" | "map_data";

type Role = "start" | "end" | "destination";

interface BuildingCoordinates {
  lat: number;
  lng: number;
}

interface Building {
  id: string;
  name: string;
  coordinates: BuildingCoordinates;
  description?: string;
  services?: Array<{
    name?: string;
    purpose?: string;
    location?: string;
  }>;
}

interface StudentSupportEntry {
  id: string;
  intent: ConciergeIntent;
  category: string;
  question_patterns: string[];
  keywords: string[];
  answer: string;
  location_ids?: string[];
  source_type: SourceType;
  last_verified_at: string;
  follow_up?: string[];
}

interface FallbackEntry {
  message: string;
  location_ids?: string[];
  follow_up?: string[];
}

interface StudentSupportKb {
  entries: StudentSupportEntry[];
  fallbacks: Record<string, FallbackEntry>;
}

interface LlmLocation {
  building_id?: string;
  building_name?: string;
  role?: Role;
}

interface LlmResult {
  message?: string;
  action?: ConciergeAction;
  locations?: LlmLocation[];
}

export interface ConciergeLocationPayload {
  building_id: string;
  building_name: string;
  coordinates: BuildingCoordinates;
  service_name: string | null;
  service_location: string | null;
  role: Role;
}

export interface ConciergeResponsePayload {
  message: string;
  locations: ConciergeLocationPayload[];
  action: ConciergeAction;
  intent: ConciergeIntent;
  verified: boolean;
  sources: string[];
  follow_up: string[];
}

export interface ConversationMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface ResolveOptions {
  allowLlm?: boolean;
  conversationHistory?: ConversationMessage[];
}

const SOURCE_PRIORITY: Record<SourceType, number> = {
  official_page: 3,
  office_confirmed: 2,
  map_data: 1,
};

const ROUTE_HINTS = [
  "route",
  "directions",
  "navigate",
  "how to get",
  "take me from",
  "from",
  "to",
  "between",
];

const LOCATION_HINTS = [
  "where is",
  "locate",
  "show me",
  "find",
  "nearest",
  "location",
  "which block",
];

const SUPPORT_HINTS = [
  "admission",
  "document",
  "registration",
  "exam",
  "transcript",
  "certificate",
  "hostel",
  "fee",
  "payment",
  "deadline",
  "complaint",
  "health",
  "clinic",
  "library",
  "borrow",
  "id card",
];

const NOTICE_HINTS = [
  "notice",
  "notices",
  "result",
  "results",
  "exam routine",
  "exam center",
  "application form",
  "summarize notice",
  "latest notice",
  "recent notice",
];

const EVENT_HINTS = [
  "event",
  "events",
  "upcoming event",
  "workshop",
  "hackathon",
  "seminar",
  "competition",
  "register for",
  "what events",
];

const CLUB_HINTS = [
  "club",
  "clubs",
  "society",
  "organization",
  "student club",
  "active club",
  "list club",
];

const LOST_FOUND_HINTS = [
  "lost and found",
  "lost item",
  "found item",
  "missing item",
  "anyone found",
  "lost something",
];

const MARKETPLACE_HINTS = [
  "book marketplace",
  "sell book",
  "buy book",
  "marketplace",
  "textbook",
  "second hand book",
  "available book",
];

const APP_HELP_HINTS = [
  "how to use",
  "app help",
  "what can you do",
  "help me",
  "what features",
  "how does this app",
];

const ACADEMIC_HINTS = [
  "semester",
  "syllabus",
  "course",
  "subject",
  "credit",
  "grade",
  "gpa",
  "cgpa",
  "backlog",
  "back exam",
  "regular exam",
  "curriculum",
  "elective",
  "project",
  "thesis",
  "internship",
  "practicals",
  "lab",
  "lecture",
  "tutorial",
  "timetable",
  "class routine",
  "faculty",
  "professor",
  "teacher",
  "hod",
  "head of department",
  "scholarship",
  "topper",
  "pass mark",
  "fail",
  "attendance",
  "absent",
  "prerequisite",
  "minor",
  "major",
  "be",
  "bachelor",
  "master",
  "msc",
  "program",
  "engineering",
  "degree",
];

const GENERAL_CAMPUS_HINTS = [
  "pulchowk",
  "campus",
  "college",
  "ioe",
  "tribhuvan",
  "tu",
  "history",
  "established",
  "founded",
  "about",
  "tell me about",
  "what is",
  "students",
  "intake",
  "branches",
  "departments",
  "affiliation",
  "accreditation",
  "ranking",
  "placement",
  "alumni",
  "campus life",
  "extracurricular",
  "sports",
  "festival",
  "annual",
  "convocation",
  "graduation",
];

const GREETING_HINTS = [
  "hello",
  "hi",
  "hey",
  "namaste",
  "good morning",
  "good afternoon",
  "good evening",
  "good night",
  "sup",
  "yo",
  "howdy",
  "greetings",
  "thanks",
  "thank you",
  "bye",
  "goodbye",
  "see you",
  "ok",
  "okay",
  "cool",
  "great",
  "nice",
  "awesome",
];

const LOCATION_ALIAS_MAP: Record<string, string> = {
  dean: "dean-office",
  "dean office": "dean-office",
  "admin building": "dean-office",
  "main office": "dean-office",
  "exam office": "exam-control-office",
  "exam control": "exam-control-office",
  library: "pulchowk-library",
  fsu: "fsu-office",
  clinic: "fsu-clinic",
  "health center": "fsu-clinic",
  canteen: "campus-canteen",
  mess: "campus-mess",
  stationery: "om-stationery",
  "print shop": "om-stationery",
  photocopy: "om-stationery",
  niraula: "niraula-stores",
  robotics: "robotics-club",
  "electrical club": "electrical-club",
  "music club": "music-club",
  seds: "seds-pulchowk",
  locus: "locus-office",
  atm: "nabil-bank-atm",
  "nabil atm": "nabil-bank-atm",
  "siddhartha atm": "siddhartha-bank-atm",
  "boys hostel": "block-a-hostel",
  "girls hostel": "girls-hostel",
  "msc hostel": "msc-hostel",
  "civil department": "dept-civil",
  "f block": "dept-civil",
  "architecture department": "dept-architecture",
  "electrical department": "dept-electrical-electronics-computer",
  "e block": "dept-electrical-electronics-computer",
  "ab block": "dept-electrical-electronics-computer",
  "computer department": "dept-electrical-electronics-computer",
  "mechanical department": "dept-mechanical-aerospace",
  "d block": "dept-mechanical-aerospace",
  "applied sciences": "dept-applied-sciences",
  humanities: "dept-science-humanities",
  gym: "gym-hall",
  "football ground": "football-ground",
  "cricket ground": "cricket-ground",
  "basketball court": "basketball-court",
  "volleyball court": "volleyball-court",
  workshop: "workshop-area",
  "main entrance": "main-entrance",
  "main gate": "main-entrance",
  gate: "main-entrance",
};

const KB = studentSupportKbData as StudentSupportKb;
const BUILDINGS = ((locationData as any)?.buildings ?? []) as Building[];
const BUILDING_BY_ID = new Map(BUILDINGS.map((item) => [item.id, item]));
const genAI = new GoogleGenerativeAI(ENV.GEMINI_API_KEY);

// ── Phase 3: In-memory TTL cache ────────────────────────────────────────────
interface CacheEntry<T> {
  data: T;
  expiresAt: number;
}
const contextCache = new Map<string, CacheEntry<string>>();
const CONTEXT_CACHE_TTL_MS = 3 * 60 * 1000; // 3 minutes

function getCachedContext(key: string): string | null {
  const entry = contextCache.get(key);
  if (entry && Date.now() < entry.expiresAt) return entry.data;
  contextCache.delete(key);
  return null;
}

function setCachedContext(key: string, data: string): void {
  contextCache.set(key, { data, expiresAt: Date.now() + CONTEXT_CACHE_TTL_MS });
}

// ── Phase 2: Fuzzy matching helper ──────────────────────────────────────────
function levenshtein(a: string, b: string): number {
  const m = a.length, n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;
  const dp: number[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = a[i-1] === b[j-1]
        ? dp[i-1][j-1]
        : 1 + Math.min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]);
    }
  }
  return dp[m][n];
}

function fuzzyIncludes(text: string, term: string, maxDistance = 2): boolean {
  if (text.includes(term)) return true;
  if (term.length < 4) return false; // skip fuzzy for very short terms
  const words = text.split(' ');
  return words.some(word => {
    if (Math.abs(word.length - term.length) > maxDistance) return false;
    return levenshtein(word, term) <= maxDistance;
  });
}

// ── Conversation history formatter ──────────────────────────────────────────
function formatConversationHistory(history: ConversationMessage[]): string {
  if (!history || history.length === 0) return '';
  const lines = history.map(m => `${m.role === 'user' ? 'Student' : 'Assistant'}: ${m.content}`);
  return `Previous conversation:\n${lines.join('\n')}\n`;
}

function normalize(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function tokenize(value: string): string[] {
  return normalize(value)
    .split(" ")
    .filter((token) => token.length > 1);
}

function includesAny(text: string, terms: string[]): boolean {
  return terms.some((term) => text.includes(term));
}

function toLocationPayload(
  building: Building,
  role: Role = "destination",
): ConciergeLocationPayload {
  return {
    building_id: building.id,
    building_name: building.name,
    coordinates: building.coordinates,
    service_name: null,
    service_location: null,
    role,
  };
}

function getBuildingSearchText(building: Building): string {
  const serviceText = (building.services ?? [])
    .flatMap((service) => [service.name, service.purpose, service.location])
    .filter(Boolean)
    .join(" ");
  return normalize(
    [building.id, building.name, building.description ?? "", serviceText].join(
      " ",
    ),
  );
}

function scoreBuildingQuery(building: Building, query: string): number {
  const normalizedQuery = normalize(query);
  const queryTokens = tokenize(query);
  const name = normalize(building.name);
  const id = normalize(building.id);
  const searchBlob = getBuildingSearchText(building);

  let score = 0;
  if (!normalizedQuery) return score;
  if (name === normalizedQuery || id === normalizedQuery) score += 100;
  if (name.includes(normalizedQuery) || normalizedQuery.includes(name))
    score += 40;

  for (const token of queryTokens) {
    if (name.includes(token)) score += 8;
    else if (id.includes(token)) score += 5;
    else if (searchBlob.includes(token)) score += 2;
  }

  return score;
}

function findAliasBuildingIds(query: string): string[] {
  const normalized = normalize(query);
  const matches = new Set<string>();
  for (const [alias, buildingId] of Object.entries(LOCATION_ALIAS_MAP)) {
    if (normalized.includes(alias)) matches.add(buildingId);
  }
  return [...matches].filter((id) => BUILDING_BY_ID.has(id));
}

function findBuildings(query: string, limit = 3): Array<{
  building: Building;
  score: number;
}> {
  const aliasIds = findAliasBuildingIds(query);
  const aliasResults = aliasIds
    .map((id) => BUILDING_BY_ID.get(id))
    .filter((value): value is Building => !!value)
    .map((building) => ({ building, score: 120 }));

  const scored = BUILDINGS.map((building) => ({
    building,
    score: scoreBuildingQuery(building, query),
  }))
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score);

  const merged = [...aliasResults];
  for (const item of scored) {
    if (merged.some((existing) => existing.building.id === item.building.id)) {
      continue;
    }
    merged.push(item);
    if (merged.length >= limit) break;
  }

  return merged.slice(0, limit);
}

function inferActionByLocationCount(count: number): ConciergeAction {
  if (count <= 1) return "show_location";
  return "show_multiple_locations";
}

function countMatches(text: string, terms: string[]): number {
  return terms.filter((term) => text.includes(term)).length;
}

function inferIntentFromQuery(query: string): ConciergeIntent {
  const normalized = normalize(query);

  // ── Route navigation (high-confidence pattern) ───────────────────────────
  if (
    normalized.includes("from ") &&
    normalized.includes(" to ") &&
    includesAny(normalized, ROUTE_HINTS)
  ) {
    return "route_navigation";
  }

  if (includesAny(normalized, ["deadline", "last date", "closing date"])) {
    return "deadline_query";
  }

  // ── App-wide intents (check BEFORE generic single-word intents) ─────────
  if (includesAny(normalized, NOTICE_HINTS)) return "notice_query";
  if (includesAny(normalized, EVENT_HINTS)) return "event_query";
  if (includesAny(normalized, CLUB_HINTS)) return "club_query";
  if (countMatches(normalized, LOST_FOUND_HINTS) >= 1) return "lost_found_query";
  if (includesAny(normalized, MARKETPLACE_HINTS)) return "marketplace_query";
  if (includesAny(normalized, APP_HELP_HINTS)) return "app_help";

  // ── Escalation: require phrase-level match to avoid "lost" false positive
  if (
    includesAny(normalized, [
      "emergency",
      "urgent",
      "complaint",
      "report harassment",
      "harassment",
    ])
  ) {
    return "escalation";
  }

  if (
    includesAny(normalized, [
      "policy",
      "rules",
      "hours",
      "timing",
      "fee",
      "payment",
      "receipt",
    ])
  ) {
    return "policy_query";
  }

  if (
    includesAny(normalized, [
      "whom to contact",
      "who handles",
      "which office",
    ]) ||
    (normalized.includes("where") && normalized.includes("office"))
  ) {
    return "office_lookup";
  }

  // ── process_howto: require "how to" / "how do" phrase or multi-keyword ──
  if (
    includesAny(normalized, ["how to", "how do", "how can", "process", "procedure"]) ||
    countMatches(normalized, ["apply", "register", "submit", "registration"]) >= 1
  ) {
    return "process_howto";
  }

  if (
    includesAny(normalized, [
      "service",
      "canteen",
      "atm",
      "water",
      "toilet",
      "hostel",
      "library",
      "clinic",
      "print",
      "stationery",
    ])
  ) {
    return "service_lookup";
  }

  if (includesAny(normalized, LOCATION_HINTS)) {
    return "location_lookup";
  }

  // ── Greeting (exact short-message match) ─────────────────────────────────
  if (includesAny(normalized, GREETING_HINTS) && normalized.split(' ').length <= 5) {
    return "greeting";
  }

  // ── Academic queries ────────────────────────────────────────────────────
  if (includesAny(normalized, ACADEMIC_HINTS)) {
    return "general_campus";
  }

  // ── General campus / college queries ─────────────────────────────────────
  if (includesAny(normalized, GENERAL_CAMPUS_HINTS)) {
    return "general_campus";
  }

  return "unknown";
}

function matchSupportEntry(query: string): StudentSupportEntry | null {
  const normalized = normalize(query);
  const candidates = KB.entries
    .map((entry) => {
      let score = 0;
      for (const pattern of entry.question_patterns ?? []) {
        const normalizedPattern = normalize(pattern);
        if (normalizedPattern && normalized.includes(normalizedPattern)) {
          score += 8;
        }
      }
      for (const keyword of entry.keywords ?? []) {
        const normalizedKeyword = normalize(keyword);
        if (normalizedKeyword && normalized.includes(normalizedKeyword)) {
          score += 2;
        } else if (normalizedKeyword && fuzzyIncludes(normalized, normalizedKeyword, 2)) {
          score += 1; // partial credit for fuzzy match
        }
      }
      return { entry, score };
    })
    .filter((item) => item.score > 0)
    .sort((a, b) => {
      if (b.score !== a.score) return b.score - a.score;
      const priorityA = SOURCE_PRIORITY[a.entry.source_type] ?? 0;
      const priorityB = SOURCE_PRIORITY[b.entry.source_type] ?? 0;
      if (priorityB !== priorityA) return priorityB - priorityA;
      return (
        new Date(b.entry.last_verified_at).getTime() -
        new Date(a.entry.last_verified_at).getTime()
      );
    });

  if (candidates.length === 0) return null;
  if (candidates[0].score < 4) return null;
  return candidates[0].entry;
}

function makeSupportResponse(entry: StudentSupportEntry): ConciergeResponsePayload {
  const locations = (entry.location_ids ?? [])
    .map((id) => BUILDING_BY_ID.get(id))
    .filter((building): building is Building => !!building)
    .map((building) => toLocationPayload(building));

  return {
    message: entry.answer,
    locations,
    action: inferActionByLocationCount(locations.length),
    intent: entry.intent,
    verified: true,
    sources: [
      `student_support_kb:${entry.id}`,
      `source_type:${entry.source_type}`,
      `last_verified:${entry.last_verified_at}`,
    ],
    follow_up: entry.follow_up ?? [],
  };
}

function isRouteQuery(query: string): boolean {
  const normalized = normalize(query);
  if (extractRouteEndpoints(query)) return true;

  return (
    includesAny(normalized, [
      "show route",
      "directions",
      "navigate",
      "how to get from",
    ])
  );
}

function extractRouteEndpoints(query: string): { start: string; end: string } | null {
  const raw = normalize(query);
  const fromTo = raw.match(/\bfrom\s+(.+?)\s+\bto\s+(.+)$/i);
  if (fromTo) {
    return {
      start: fromTo[1].trim(),
      end: fromTo[2].trim(),
    };
  }

  const betweenAnd = raw.match(/\bbetween\s+(.+?)\s+\band\s+(.+)$/i);
  if (betweenAnd) {
    return {
      start: betweenAnd[1].trim(),
      end: betweenAnd[2].trim(),
    };
  }

  // Support terse route queries such as: "dean office to om stationery store"
  const plainTo = raw.match(/^(.+?)\s+\bto\s+(.+)$/i);
  if (plainTo) {
    const start = plainTo[1].trim();
    const end = plainTo[2].trim();
    const invalidPlainStart = new Set([
      "how",
      "what",
      "where",
      "when",
      "why",
      "who",
      "go",
      "walk",
      "move",
      "head",
      "get",
      "take",
      "need",
      "want",
      "trying",
    ]);
    const startLower = start.toLowerCase();
    const isQuestionOrVerbPrefix = 
      invalidPlainStart.has(startLower) ||
      startLower.startsWith("how ") ||
      startLower.startsWith("what ") ||
      startLower.startsWith("where ") ||
      startLower.startsWith("when ") ||
      startLower.startsWith("why ") ||
      startLower.startsWith("who ") ||
      startLower.startsWith("can ") ||
      startLower.startsWith("could ") ||
      startLower.startsWith("should ") ||
      startLower.startsWith("is ") ||
      startLower.startsWith("are ") ||
      startLower.startsWith("tell ") ||
      startLower.startsWith("want ") ||
      startLower.startsWith("need ");

    if (!isQuestionOrVerbPrefix && start.length > 2 && end.length > 2) {
      return { start, end };
    }
  }

  return null;
}

function resolveDeterministicRoute(query: string): ConciergeResponsePayload | null {
  const endpoints = extractRouteEndpoints(query);
  if (!endpoints) return null;

  const startMatch = findBuildings(endpoints.start, 1)[0];
  const endMatch = findBuildings(endpoints.end, 1)[0];
  if (!startMatch || !endMatch) return null;
  if (startMatch.score < 10 || endMatch.score < 10) return null;
  if (startMatch.building.id === endMatch.building.id) return null;

  const locations: ConciergeLocationPayload[] = [
    toLocationPayload(startMatch.building, "start"),
    toLocationPayload(endMatch.building, "end"),
  ];

  return {
    message: `From ${startMatch.building.name}, walk toward ${endMatch.building.name}. I have pinned both points so you can follow the map route clearly.`,
    locations,
    action: "show_route",
    intent: "route_navigation",
    verified: true,
    sources: ["campus_data:buildings", "source_type:map_data"],
    follow_up: [
      "If you want, ask for landmarks near the destination before you start.",
    ],
  };
}

function isLocationLookupQuery(query: string): boolean {
  const normalized = normalize(query);
  return (
    includesAny(normalized, LOCATION_HINTS) ||
    normalized.startsWith("where ") ||
    normalized.startsWith("show ")
  );
}

function resolveLocationLookup(query: string): ConciergeResponsePayload | null {
  const matches = findBuildings(query, 3).filter((item) => item.score >= 10);
  if (matches.length === 0) return null;

  const locations = matches.map((match) => toLocationPayload(match.building));
  if (locations.length === 1) {
    const building = matches[0].building;
    return {
      message: `${building.name} is mapped on campus. ${building.description ?? "I have highlighted the location for you."}`,
      locations,
      action: "show_location",
      intent: "location_lookup",
      verified: true,
      sources: ["campus_data:buildings", "source_type:map_data"],
      follow_up: ["Ask for directions from your current point if needed."],
    };
  }

  return {
    message: `I found multiple relevant locations: ${matches.map((item) => item.building.name).join(", ")}.`,
    locations,
    action: "show_multiple_locations",
    intent: "location_lookup",
    verified: true,
    sources: ["campus_data:buildings", "source_type:map_data"],
    follow_up: ["Tell me your start and destination if you want route guidance."],
  };
}

// ── Intents that should show a physical location on the map when falling back ──
const NAVIGATIONAL_FALLBACK_INTENTS = new Set<ConciergeIntent>([
  "route_navigation",
  "location_lookup",
  "service_lookup",
  "office_lookup",
  "escalation",
]);

function getFallbackLocationIdsByIntent(
  intent: ConciergeIntent,
  query: string,
): string[] {
  const normalized = normalize(query);
  if (
    includesAny(normalized, ["exam", "admit", "result", "transcript", "roll"])
  ) {
    return ["exam-control-office"];
  }
  if (includesAny(normalized, ["health", "medical", "sick", "injury"])) {
    return ["fsu-clinic"];
  }
  if (includesAny(normalized, ["complaint", "harassment", "union"])) {
    return ["fsu-office"];
  }
  if (includesAny(normalized, ["hostel", "mess", "canteen"])) {
    return ["campus-mess", "girls-hostel"];
  }
  if (includesAny(normalized, ["library", "book issue", "reading"])) {
    return ["pulchowk-library"];
  }
  if (includesAny(normalized, ["print", "stationery", "photocopy"])) {
    return ["om-stationery"];
  }
  if (intent === "deadline_query") {
    return ["exam-control-office", "dean-office"];
  }
  if (intent === "escalation") {
    return ["dean-office"];
  }
  // No default dean-office for non-navigational intents
  return [];
}

function buildFallbackResponse(
  intent: ConciergeIntent,
  query: string,
): ConciergeResponsePayload {
  // For non-navigational intents, return a text-only response (no map pin)
  if (!NAVIGATIONAL_FALLBACK_INTENTS.has(intent)) {
    return {
      message: "I'm not sure about that yet, but I'm here to help with anything related to Pulchowk Campus — notices, events, clubs, navigation, academics, and more. Try asking something specific!",
      locations: [],
      action: "text_answer",
      intent,
      verified: false,
      sources: [],
      follow_up: [
        "Summarize recent notices",
        "Any upcoming events?",
        "Where is the library?",
      ],
    };
  }

  const fallbackKey =
    intent === "deadline_query" ||
    intent === "escalation" ||
    intent === "policy_query"
      ? intent
      : "general";
  const fallback = KB.fallbacks[fallbackKey] ?? KB.fallbacks.general;
  const locationIds = fallback.location_ids?.length
    ? fallback.location_ids
    : getFallbackLocationIdsByIntent(intent, query);

  const locations = locationIds
    .map((id) => BUILDING_BY_ID.get(id))
    .filter((building): building is Building => !!building)
    .map((building) => toLocationPayload(building));

  // If no locations resolved, return text-only
  if (locations.length === 0) {
    return {
      message: fallback.message,
      locations: [],
      action: "text_answer",
      intent,
      verified: false,
      sources: [],
      follow_up:
        fallback.follow_up ?? [
          "Ask a specific question and I will try to help.",
        ],
    };
  }

  return {
    message: fallback.message,
    locations,
    action: inferActionByLocationCount(locations.length),
    intent: intent === "unknown" ? "office_lookup" : intent,
    verified: false,
    sources: [],
    follow_up:
      fallback.follow_up ?? [
        "Ask a specific office-focused question and I will guide you there.",
      ],
  };
}

function safeJsonParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    const start = text.indexOf("{");
    const end = text.lastIndexOf("}");
    if (start >= 0 && end > start) {
      try {
        return JSON.parse(text.slice(start, end + 1));
      } catch {
        return null;
      }
    }
    return null;
  }
}

function sanitizeAction(action: unknown): ConciergeAction {
  if (action === "show_route") return action;
  if (action === "show_multiple_locations") return action;
  if (action === "text_answer") return action;
  return "show_location";
}

function mapLlmLocation(raw: LlmLocation, role: Role): ConciergeLocationPayload | null {
  const byId =
    typeof raw.building_id === "string" ? BUILDING_BY_ID.get(raw.building_id) : null;
  const byName =
    !byId && typeof raw.building_name === "string"
      ? findBuildings(raw.building_name, 1)[0]?.building
      : null;
  const resolved = byId ?? byName;
  if (!resolved) return null;
  const locationRole: Role = raw.role === "start" || raw.role === "end" ? raw.role : role;
  return toLocationPayload(resolved, locationRole);
}

async function resolveWithLlmNavigation(
  query: string,
  history: ConversationMessage[] = [],
): Promise<ConciergeResponsePayload | null> {
  try {
    const model = genAI.getGenerativeModel({
      model: "gemini-2.5-flash",
      generationConfig: { responseMimeType: "application/json" },
    });

    // Pre-filter buildings: top 10 most relevant + any alias matches
    const aliasIds = findAliasBuildingIds(query);
    const scored = BUILDINGS.map((b) => ({ b, s: scoreBuildingQuery(b, query) }))
      .sort((a, b) => b.s - a.s);
    const seen = new Set<string>();
    const relevantBuildings: Array<{ id: string; name: string; coordinates: BuildingCoordinates; description: string }> = [];
    for (const aid of aliasIds) {
      const b = BUILDING_BY_ID.get(aid);
      if (b && !seen.has(b.id)) {
        seen.add(b.id);
        relevantBuildings.push({ id: b.id, name: b.name, coordinates: b.coordinates, description: b.description ?? "" });
      }
    }
    for (const { b } of scored) {
      if (relevantBuildings.length >= 10) break;
      if (!seen.has(b.id)) {
        seen.add(b.id);
        relevantBuildings.push({ id: b.id, name: b.name, coordinates: b.coordinates, description: b.description ?? "" });
      }
    }

    const historyBlock = formatConversationHistory(history);

    const prompt = `You are a campus navigation assistant for Pulchowk Campus.
Use only the provided verified building list.
Do not invent buildings.
${historyBlock ? `\n${historyBlock}\n` : ''}
Buildings:
${JSON.stringify(relevantBuildings)}

User query: ${query}

Instructions:
- If the user refers to something from the previous conversation (e.g. "there", "that place", "it"), resolve it using the conversation context.
- Provide a short, helpful response.

Return only JSON:
{
  "message": "short helpful text without coordinates",
  "action": "show_route" | "show_location" | "show_multiple_locations",
  "locations": [
    {
      "building_id": "verified id",
      "building_name": "verified name",
      "role": "start" | "end" | "destination"
    }
  ]
}`;

    const result = await model.generateContent(prompt);
    const parsed = safeJsonParse(result.response.text()) as LlmResult | null;
    if (!parsed || !Array.isArray(parsed.locations)) return null;

    const action = sanitizeAction(parsed.action);
    const mappedLocations = parsed.locations
      .map((item, index) =>
        mapLlmLocation(item, action === "show_route" && index === 0 ? "start" : action === "show_route" ? "end" : "destination"),
      )
      .filter((item): item is ConciergeLocationPayload => !!item);

    if (action === "show_route" && mappedLocations.length < 2) return null;
    if (action !== "show_route" && mappedLocations.length < 1) return null;

    return {
      message:
        typeof parsed.message === "string" && parsed.message.trim().length > 0
          ? parsed.message.trim()
          : "I found relevant campus locations and highlighted them on the map.",
      action,
      locations: mappedLocations,
      intent: action === "show_route" ? "route_navigation" : "location_lookup",
      verified: true,
      sources: ["campus_data:buildings", "source_type:map_data", "llm:gemini-2.5-flash"],
      follow_up: ["Ask for landmark-based directions if you need simpler wayfinding."],
    };
  } catch (error) {
    console.error("LLM navigation fallback failed:", error);
    return null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// General-purpose LLM resolver for college questions
// ─────────────────────────────────────────────────────────────────────────────

const CAMPUS_KNOWLEDGE = `
Pulchowk Campus (officially: IOE Pulchowk Campus) is the central and premier engineering campus under the
Institute of Engineering (IOE), Tribhuvan University (TU), Nepal. It is located in Lalitpur, Kathmandu Valley.

Key facts:
- Full name: Institute of Engineering, Pulchowk Campus
- Established: 1972 (as Nepal Engineering Institute), became part of TU in 1984
- Affiliation: Tribhuvan University, Institute of Engineering
- Location: Pulchowk, Lalitpur, Nepal
- Programs offered:
  - Bachelor of Engineering (BE) — 4 years (8 semesters)
    - Civil Engineering
    - Computer Engineering
    - Electronics, Communication & Information Engineering
    - Electrical Engineering
    - Mechanical Engineering
    - Aerospace Engineering
    - Architecture (B.Arch — 5 years)
  - Master of Science (M.Sc.) in various engineering fields
  - PhD programs
- Admission: Through IOE Entrance Examination conducted by IOE
- Semester system: 8 semesters for BE, each ~5 months
- Grading: Internal assessments + final board exams; results published at exam.ioe.edu.np
- Pass marks: 40% in each subject
- Backlog policy: Students can carry failed subjects and retake in subsequent exams
- Student clubs: LOCUS, Robotics Club, SEDS, Electrical Club, Music Club, etc.
- Annual events: LOCUS (tech fest), Yantra (mechanical fest), sports week
- Facilities: Library, computer labs (ICTC), workshops, hostels, canteen, sports grounds, gym
- Campus head: Dean of IOE (administrative head of all IOE campuses)
`.trim();

async function resolveWithGeneralLlm(
  query: string,
  intent: ConciergeIntent,
  history: ConversationMessage[] = [],
): Promise<ConciergeResponsePayload | null> {
  try {
    const model = genAI.getGenerativeModel({
      model: "gemini-2.5-flash",
      generationConfig: { responseMimeType: "application/json" },
    });

    const historyBlock = formatConversationHistory(history);

    const prompt = `You are Smart Pulchowk Assistant — a helpful, knowledgeable campus companion for Pulchowk Campus (IOE, Tribhuvan University) students.

About Pulchowk Campus:
${CAMPUS_KNOWLEDGE}

About the Smart Pulchowk app:
${APP_KNOWLEDGE}
${historyBlock ? `\n${historyBlock}` : ''}
User query: ${query}

Instructions:
- Answer the student's question helpfully and accurately using the knowledge provided.
- If the user refers to something from the previous conversation, resolve it using context.
- For greetings, respond warmly and suggest what you can help with.
- For campus-related questions, provide accurate information from the knowledge above.
- For questions you genuinely cannot answer (not related to campus/college at all), politely say so and suggest campus-related topics you can help with.
- Do NOT make up specific numbers, dates, or facts that are not in the context.
- Do NOT reference any physical location or suggest going to an office unless the question is specifically about visiting an office.
- Use markdown formatting: **bold** for emphasis, bullet lists with - for steps.
- Keep responses concise but informative.
- Suggest 1-3 short follow-up questions the student might ask next.

Return only JSON:
{
  "message": "your helpful response",
  "follow_up": ["follow-up question 1", "follow-up question 2"]
}`;

    const result = await model.generateContent(prompt);
    const parsed = safeJsonParse(result.response.text()) as {
      message?: string;
      follow_up?: string[];
    } | null;

    if (
      !parsed ||
      typeof parsed.message !== "string" ||
      !parsed.message.trim()
    ) {
      return null;
    }

    const followUp = Array.isArray(parsed.follow_up)
      ? parsed.follow_up
          .filter((s): s is string => typeof s === 'string' && s.trim().length > 0)
          .slice(0, 3)
      : [];

    return {
      message: parsed.message.trim(),
      locations: [],
      action: "text_answer",
      intent,
      verified: false,
      sources: ["campus_knowledge", "llm:gemini-2.5-flash"],
      follow_up: followUp,
    };
  } catch (error) {
    console.error("General LLM resolution failed:", error);
    return null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// App-wide context resolution (hybrid: live DB data + LLM)
// ─────────────────────────────────────────────────────────────────────────────

const APP_INTENT_TOPICS: Partial<Record<ConciergeIntent, AppContextTopic>> = {
  notice_query: "notices",
  event_query: "events",
  club_query: "clubs",
  lost_found_query: "lost_found",
  marketplace_query: "marketplace",
  app_help: "all",
};

const APP_KNOWLEDGE = `
Smart Pulchowk is a campus companion app for Pulchowk Campus (IOE, Tribhuvan University) students.
Key features:
- Notices: IOE exam results, routines, application forms, and general campus notices (auto-synced from exam.ioe.tu.edu.np).
- Events: Browse and register for campus events organized by student clubs.
- Clubs: View active student clubs, their profiles, missions, and social links.
- Map & Navigation: Interactive campus map with building locations and route guidance.
- Book Marketplace: Buy and sell second-hand textbooks with other students.
- Lost & Found: Report or search for lost/found items on campus.
- Chat: Direct messaging between students for book deals.
- Notifications: Push notifications for new notices, events, and chat messages.
- Calendar: Academic calendar view.
- Classroom: Classroom and course information.
`.trim();

// ── Per-topic FAQ sections for selective injection ─────────────────────────
const FAQ_SECTIONS: Record<string, string> = {
  marketplace: `MARKETPLACE FAQ:
- Q: How do I sell a book? A: Go to Marketplace tab -> "Sell a Book". Fill title, author, price, condition, upload photos.
- Q: How do I send a purchase request? A: Open listing -> "Request to Buy" -> optional message.
- Q: Where are my sent requests? A: Activity icon (top-right) or Marketplace Activity -> "Requests" tab.
- Q: Where are requests I've received? A: Marketplace Activity -> "Inquiries" tab.
- Q: How to accept/reject request? A: Marketplace Activity -> Inquiries -> "Accept" or "Reject".
- Q: How to contact seller? A: After request is accepted, "Contact Seller" button appears on listing/Requests tab.
- Q: Where are saved books? A: Marketplace Activity -> "Saved" tab.
- Q: How to delete a chat? A: Long-press conversation in chat list -> trash icon.
- Q: How to delete my listing? A: Marketplace Activity -> Selling -> Find listing -> delete icon.
- Q: How to mark as sold? A: Marketplace Activity -> Selling -> "Mark as Sold".
- Q: How to cancel a sent request? A: Marketplace Activity -> Requests -> Find pending request -> "Cancel".
- Q: How to rate a seller? A: After request is accepted and completed, "Rate Seller" button appears in Requests tab.
- Q: Can I report/block users? A: Yes, via the 3-dot menu (⋮) on listing/profile.`,

  classroom: `CLASSROOM FAQ:
- Q: How setup profile? A: Done automatically based on college email; displays subjects/assignments for your semester.
- Q: How do I submit an assignment? A: Classroom tab -> Find assignment -> "Upload File" (PDF/Image) -> "Submit Assignment".
- Q: Can I resubmit? A: Yes, before grading.
- Q: How to check grades? A: Look for "Returned/Graded" status in Classroom list; tap for feedback.
- Q: How to add subject (Teachers)? A: Classroom -> "Add Subject" tab -> Select Faculty/Semester/Subject.
- Q: How to create assignment (Teachers)? A: Classroom -> "Assign" tab -> Select subject/category/due date.
- Q: How to view submissions (Teachers)? A: Classroom -> "Managed" tab -> tap assignment to see student list.
- Q: Data not loading? A: Pull down on Classroom screen to refresh/sync.`,

  notices: `NOTICES FAQ:
- Q: Dept notices? A: Automatically filtered by faculty/semester from your email.
- Q: Search old notices? A: Use search bar at top of Notices tab.
- Q: Important notices? A: Highlighted with border or pinned to top.
- Q: Push notifications for notices? A: Enable "Notices" in Settings -> Notification Settings.
- Q: Download attachments? A: Open notice -> tap thumbnail to view/save PDF/image.`,

  events: `EVENTS FAQ:
- Q: Register for event? A: Tap event -> "Register" or "Join" button at bottom.
- Q: My events? A: See "My Events" or calendar in Events tab.
- Q: Sync to device calendar? A: Event details -> "Add to Calendar".
- Q: See all clubs events? A: Events tab provides a unified feed for all clubs.
- Q: Directions to venue? A: Tap venue name to open Google Maps.`,

  clubs: `CLUBS FAQ:
- Q: How to join? A: Club profile -> "Join" (if recruiting) or use contact links.
- Q: List of all clubs? A: Main "Clubs" tab lists all registered orgs.
- Q: Contact club? A: Use Email, WhatsApp, or Social buttons on club profile.
- Q: Create new club? A: Must register with admin; student council can guide you.
- Q: See club activities? A: Visit club profile or main Events feed.`,

  lost_found: `LOST & FOUND FAQ:
- Q: Report found item? A: Lost & Found tab -> "Report" -> "I found something".
- Q: Find lost item? A: Browse "Lost" tab; use search/category filters.
- Q: Claim item? A: Tap item -> "Claim Item" to contact finder (verify ownership first).
- Q: Privacy? A: Phone number only shared with the person you coordinate return with.
- Q: Manage my reports? A: Profile icon (top right) in Lost & Found -> "My Items".`,

  map: `MAP FAQ:
- Q: Search building/room? A: Top search bar -> enter name (e.g., "Electrical Block") or room ID.
- Q: Icon meanings? A: Blue=Academic, Orange=Food, Sports=Athletic. Tap for details.
- Q: Walking directions? A: Tap marker -> "Directions". Uses GPS or manual start point.
- Q: Satellite view? A: "Layers" icon -> toggle Satellite view.`,

  notifications: `NOTIFICATIONS FAQ:
- Q: Not receiving alerts? A: Check app Settings and phone system permissions.
- Q: Mark as read? A: Tap to view or swipe right on list item.
- Q: Delete notification? A: Swipe left on list item (red background).
- Q: Choose alerts? A: Settings -> Notification Settings; toggle Books, Notices, Events, etc.`,
};

const INTENT_TO_FAQ_KEYS: Partial<Record<ConciergeIntent, string[]>> = {
  notice_query: ["notices"],
  event_query: ["events"],
  club_query: ["clubs"],
  lost_found_query: ["lost_found"],
  marketplace_query: ["marketplace"],
};

function getRelevantFaq(intent: ConciergeIntent): string {
  const keys = INTENT_TO_FAQ_KEYS[intent];
  if (keys) {
    return keys.map((k) => FAQ_SECTIONS[k] ?? "").filter(Boolean).join("\n\n");
  }
  // For app_help / unknown, return all FAQ sections
  return Object.values(FAQ_SECTIONS).join("\n\n");
}

async function resolveWithAppContext(
  query: string,
  intent: ConciergeIntent,
  history: ConversationMessage[] = [],
): Promise<ConciergeResponsePayload | null> {
  const topic = APP_INTENT_TOPICS[intent];
  if (!topic) return null;

  try {
    // Phase 3: Use cached context if available
    const cacheKey = `context_${topic}`;
    let contextData = getCachedContext(cacheKey);
    if (!contextData) {
      contextData = await buildAppContext(topic);
      setCachedContext(cacheKey, contextData);
    }

    const model = genAI.getGenerativeModel({
      model: "gemini-2.5-flash",
      generationConfig: { responseMimeType: "application/json" },
    });

    const relevantFaq = getRelevantFaq(intent);
    const historyBlock = formatConversationHistory(history);

    const prompt = `You are Smart Pulchowk Assistant — a smart, friendly campus companion for Pulchowk Campus students.

About the app:
${APP_KNOWLEDGE}

App FAQ (How-to guidance):
${relevantFaq}

Live data from the app:
${contextData}
${historyBlock ? `\n${historyBlock}` : ''}
User query: ${query}

Instructions:
- Answer the user's question using the live data and FAQs provided.
- Be concise, helpful, and student-friendly.
- If the user refers to something from the previous conversation (e.g. "those", "more details", "first one"), resolve it using the conversation context.
- If the user asks "how-to" questions, refer to the FAQ section.
- If the user asks to summarize notices, provide a clear summary grouped by category.
- If the user asks about events, list them with dates and organizers.
- If no relevant data is available, say so politely and suggest alternatives.
- Do NOT make up data that is not in the context above.
- Use markdown formatting: **bold** for emphasis, bullet lists with - for steps.
- Suggest 1-3 short follow-up questions the student might ask next.

Return only JSON:
{
  "message": "your helpful response",
  "follow_up": ["follow-up question 1", "follow-up question 2"]
}`;

    const result = await model.generateContent(prompt);
    const parsed = safeJsonParse(result.response.text()) as {
      message?: string;
      follow_up?: string[];
    } | null;

    if (
      !parsed ||
      typeof parsed.message !== "string" ||
      !parsed.message.trim()
    ) {
      return null;
    }

    // Phase 2: Extract follow-up suggestions from LLM
    const followUp = Array.isArray(parsed.follow_up)
      ? parsed.follow_up
          .filter((s): s is string => typeof s === 'string' && s.trim().length > 0)
          .slice(0, 3)
      : [];

    return {
      message: parsed.message.trim(),
      locations: [],
      action: "text_answer",
      intent,
      verified: true,
      sources: [`app_context:${topic}`, "llm:gemini-2.5-flash"],
      follow_up: followUp,
    };
  } catch (error) {
    console.error("App context resolution failed:", error);
    return null;
  }
}

export async function resolveStudentConciergeQuery(
  query: string,
  options: ResolveOptions = {},
): Promise<ConciergeResponsePayload> {
  const rawQuery = query.trim();
  const normalizedQuery = normalize(rawQuery);
  const allowLlm = options.allowLlm ?? true;
  const history = options.conversationHistory ?? [];

  if (!rawQuery) {
    return buildFallbackResponse("unknown", rawQuery);
  }

  // ── Route navigation ────────────────────────────────────────────────────
  if (isRouteQuery(normalizedQuery)) {
    const deterministic = resolveDeterministicRoute(rawQuery);
    if (deterministic) return deterministic;

    if (allowLlm) {
      const llm = await resolveWithLlmNavigation(rawQuery, history);
      if (llm) return llm;
    }

    return buildFallbackResponse("route_navigation", rawQuery);
  }

  // ── Knowledge base match ────────────────────────────────────────────────
  const matchedEntry = matchSupportEntry(rawQuery);
  if (matchedEntry) {
    return makeSupportResponse(matchedEntry);
  }

  // ── Explicit location lookup ────────────────────────────────────────────
  if (isLocationLookupQuery(rawQuery)) {
    const locationLookup = resolveLocationLookup(rawQuery);
    if (locationLookup) return locationLookup;
  }

  const inferredIntent = inferIntentFromQuery(rawQuery);

  // ── Greetings: handle with general LLM (no map) ─────────────────────────
  if (inferredIntent === "greeting" && allowLlm) {
    const greeting = await resolveWithGeneralLlm(rawQuery, inferredIntent, history);
    if (greeting) return greeting;
    // Simple fallback greeting if LLM fails
    return {
      message: "Hello! 👋 I'm your Smart Pulchowk Assistant. I can help you with campus navigation, notices, events, clubs, academics, and more. What would you like to know?",
      locations: [],
      action: "text_answer",
      intent: "greeting",
      verified: true,
      sources: [],
      follow_up: [
        "Summarize recent notices",
        "Any upcoming events?",
        "Where is the library?",
      ],
    };
  }

  // ── App-wide intent resolution (hybrid: live DB + LLM) ──────────────────
  const isAppIntent = inferredIntent in APP_INTENT_TOPICS;
  if (isAppIntent && allowLlm) {
    const appResult = await resolveWithAppContext(rawQuery, inferredIntent, history);
    if (appResult) return appResult;
  }

  // ── General campus / academic queries: use general LLM ──────────────────
  if (inferredIntent === "general_campus" && allowLlm) {
    const campusResult = await resolveWithGeneralLlm(rawQuery, inferredIntent, history);
    if (campusResult) return campusResult;
  }

  // ── Support-heavy intents: use KB fallback (may show office location) ───
  const supportHeavy =
    inferredIntent === "process_howto" ||
    inferredIntent === "policy_query" ||
    inferredIntent === "office_lookup" ||
    inferredIntent === "deadline_query" ||
    inferredIntent === "escalation" ||
    includesAny(normalizedQuery, SUPPORT_HINTS);

  if (supportHeavy) {
    return buildFallbackResponse(inferredIntent, rawQuery);
  }

  // ── Location lookup as secondary pass ───────────────────────────────────
  const locationLookup = resolveLocationLookup(rawQuery);
  if (locationLookup) return locationLookup;

  if (isLocationLookupQuery(rawQuery) && allowLlm) {
    const llm = await resolveWithLlmNavigation(rawQuery, history);
    if (llm) return llm;
  }

  // ── Final fallback: try general LLM for ANY query before giving up ──────
  if (allowLlm) {
    // First try app context (for data-backed answers)
    const appFallback = await resolveWithAppContext(rawQuery, "app_help", history);
    if (appFallback) return appFallback;

    // Then try general campus LLM (for broader college questions)
    const generalFallback = await resolveWithGeneralLlm(rawQuery, inferredIntent, history);
    if (generalFallback) return generalFallback;
  }

  // ── Absolute last resort: text-only fallback (no dean-office map pin) ───
  return buildFallbackResponse(inferredIntent, rawQuery);
}
